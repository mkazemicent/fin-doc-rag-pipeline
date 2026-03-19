import hashlib
import logging
from typing import Optional

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from src.ingestion.hash_tracker import IngestionTracker
from src.config import Settings, get_settings
from src.rag.utils import size_cap_chunk

logger = logging.getLogger(__name__)


class ChromaDealStore:
    """
    Domain-specific wrapper for ChromaDB, handling financial document storage and retrieval.
    Uses Client-Server architecture to avoid file-locking conflicts.
    """
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

        # 1. Initialize Embeddings
        self.embeddings = OllamaEmbeddings(
            model=self.settings.embedding_model,
            base_url=self.settings.ollama_base_url,
            num_ctx=self.settings.num_ctx,
        )

        # 2. Connect to the ChromaDB Service (Client-Server Mode)
        logger.info(f"Connecting to ChromaDB Server at {self.settings.chroma_host}:{self.settings.chroma_port}")

        self.client = chromadb.HttpClient(
            host=self.settings.chroma_host,
            port=self.settings.chroma_port
        )

        # 3. Initialize the LangChain Chroma wrapper using the remote client
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.settings.collection_name,
            embedding_function=self.embeddings
        )
        self.tracker_path = str(self.settings.hash_db_path)

    def get_retriever(self, k: Optional[int] = None, where_filter: Optional[dict] = None) -> VectorStoreRetriever:
        k = k or self.settings.retriever_k
        search_kwargs = {
            "k": k,
            "fetch_k": k * self.settings.fetch_k_multiplier,
            "lambda_mult": self.settings.mmr_lambda,
        }
        if where_filter:
            search_kwargs["filter"] = where_filter
        return self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs,
        )

    def initialize_deal_store(self, access_group: str = "general") -> None:
        """
        Reads processed .txt files, chunks them, and stores them in the remote ChromaDB.
        Processes each file independently so a failed batch never marks a file as complete.
        Uses deterministic chunk IDs to prevent duplication on re-ingestion.
        """
        processed_dir = self.settings.processed_data_dir

        logger.info("=====================================================")
        logger.info("Initializing ChromaDealStore via Server Connection")
        logger.info(f"Using Tracker at: {self.tracker_path}")
        logger.info("=====================================================")

        if not processed_dir.exists():
            logger.error(f"Processed data directory not found at {processed_dir}")
            return

        with IngestionTracker(self.tracker_path) as tracker:
            txt_files = list(processed_dir.glob("*.txt"))
            if not txt_files:
                logger.warning("No .txt files found to process.")
                return

            # Two-stage chunking:
            # Stage 1: SemanticChunker detects topic boundaries via embeddings
            # Stage 2: size_cap_chunk() caps oversized semantic chunks
            semantic_chunker = SemanticChunker(
                self.embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=self.settings.semantic_threshold,
            )

            total_chunks = 0
            files_processed = 0

            for txt_file in txt_files:
                normalized_name = txt_file.stem + ".pdf"

                already_processed, file_hash = tracker.check_and_hash(str(txt_file))
                if already_processed:
                    continue

                # Load and tag the document
                loader = TextLoader(str(txt_file), encoding="utf-8")
                loaded_docs = loader.load()

                for doc in loaded_docs:
                    doc.metadata["source"] = normalized_name
                    doc.metadata["access_group"] = access_group

                # Two-stage chunking: semantic split → size-cap fallback
                # TextLoader produces one Document per file, so we pre-batch
                # the text into paragraph-sized segments (≤6000 chars) before
                # calling SemanticChunker. This keeps each embedding call within
                # nomic-embed-text's 8192-token context window on any size document.
                _MAX_SEMANTIC_CHARS = 6000
                metadata = loaded_docs[0].metadata
                full_text = loaded_docs[0].page_content

                # Batch paragraphs into segments within the model's context limit
                paragraphs = [p for p in full_text.split('\n') if p.strip()]
                segments: list[str] = []
                buffer: list[str] = []
                buffer_len = 0
                for para in paragraphs:
                    if buffer_len + len(para) > _MAX_SEMANTIC_CHARS and buffer:
                        segments.append('\n'.join(buffer))
                        buffer, buffer_len = [], 0
                    buffer.append(para)
                    buffer_len += len(para) + 1
                if buffer:
                    segments.append('\n'.join(buffer))

                semantic_texts = []
                for segment in segments:
                    try:
                        semantic_texts.extend(semantic_chunker.split_text(segment))
                    except Exception as e:
                        logger.warning(f"SemanticChunker failed on segment of {normalized_name}, using segment as-is: {e}")
                        semantic_texts.append(segment)

                if not semantic_texts:
                    logger.warning(f"No text extracted from {normalized_name}, skipping.")
                    continue

                chunked_documents = size_cap_chunk(
                    semantic_texts, metadata,
                    max_chunk_size=self.settings.max_chunk_size,
                    chunk_size=self.settings.chunk_size,
                    chunk_overlap=self.settings.chunk_overlap,
                    min_chunk_chars=self.settings.min_chunk_chars,
                )

                logger.info(f"Created {len(chunked_documents)} chunks from {normalized_name} (semantic + size-cap).")

                # Generate deterministic IDs for each chunk
                chunk_ids = [
                    hashlib.sha256(f"{normalized_name}::chunk::{idx}".encode()).hexdigest()
                    for idx in range(len(chunked_documents))
                ]

                # Batch insert with per-file failure tracking
                batch_size = self.settings.batch_size
                file_failed = False

                for i in range(0, len(chunked_documents), batch_size):
                    batch = chunked_documents[i : i + batch_size]
                    batch_ids = chunk_ids[i : i + batch_size]
                    try:
                        self.vectorstore.add_documents(documents=batch, ids=batch_ids)
                        logger.info(f"Successfully sent batch {(i // batch_size) + 1} for {normalized_name}.")
                    except Exception as e:
                        logger.error(f"Failed to send batch for {normalized_name}: {e}")
                        file_failed = True

                # Only mark as processed if ALL batches for this file succeeded
                if not file_failed:
                    tracker.mark_as_processed_with_hash(str(txt_file), file_hash)
                    total_chunks += len(chunked_documents)
                    files_processed += 1
                else:
                    logger.error(f"Skipping tracker update for {normalized_name} due to batch failure — will retry on next run.")

            logger.info(f"Finished indexing: {files_processed} files, {total_chunks} chunks.")

    def delete_deal_document(self, filename: str) -> int:
        """Purges a document from the remote store and tracker."""
        with IngestionTracker(self.tracker_path) as tracker:
            matches = self.vectorstore.get(where={"source": filename})
            matching_ids = matches.get("ids", [])
            if matching_ids:
                self.vectorstore.delete(ids=matching_ids)
                tracker.remove_from_tracker(filename)
                return len(matching_ids)
            return 0

    def reset_collection(self) -> None:
        """Drops and recreates the collection for re-ingestion after chunking changes."""
        logger.info(f"Resetting collection '{self.settings.collection_name}' and ingestion tracker.")
        self.client.delete_collection(self.settings.collection_name)
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.settings.collection_name,
            embedding_function=self.embeddings
        )
        with IngestionTracker(self.tracker_path) as tracker:
            tracker.reset()
        logger.info("Collection and tracker reset complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    ChromaDealStore().initialize_deal_store()
