import os
import logging
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ingestion.hash_tracker import IngestionTracker

# Load environment variables
load_dotenv('.env.local')

# Configure standard Python logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- ROBUST PATH RESOLUTION VIA ENV ---
# Prioritize the DATA_DIR env variable set in docker-compose.yml
DATA_ROOT_STR = os.getenv("DATA_DIR")
if DATA_ROOT_STR:
    DATA_ROOT = Path(DATA_ROOT_STR)
else:
    # Fallback for local development
    DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"

PROCESSED_DATA_DIR = DATA_ROOT / "processed"
CHROMA_DB_DIR = DATA_ROOT / "chroma_db"
HASH_DB_PATH = DATA_ROOT / "ingestion_state.db"

class ChromaDealStore:
    """
    Domain-specific wrapper for ChromaDB, handling financial document storage and retrieval.
    Now uses Client-Server architecture to avoid file-locking conflicts.
    """
    def __init__(self):
        # 1. Initialize Embeddings
        self.embeddings = OllamaEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "mxbai-embed-large"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        
        # 2. Connect to the ChromaDB Service (Client-Server Mode)
        self.host = os.getenv("CHROMA_HOST", "localhost")
        self.port = int(os.getenv("CHROMA_PORT", "8000"))
        
        logger.info(f"Connecting to ChromaDB Server at {self.host}:{self.port}")
        
        # Initialize the HTTP Client
        self.client = chromadb.HttpClient(host=self.host, port=self.port)
        
        # Initialize the LangChain Chroma wrapper using the remote client
        self.vectorstore = Chroma(
            client=self.client,
            collection_name="deal_documents",
            embedding_function=self.embeddings
        )
        self.tracker_path = str(HASH_DB_PATH)

    def get_retriever(self, k: int = 8):
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def initialize_deal_store(self):
        """
        Reads processed .txt files, chunks them, and stores them in the remote ChromaDB.
        """
        logger.info("=====================================================")
        logger.info("Initializing ChromaDealStore via Server Connection")
        logger.info(f"Using Tracker at: {self.tracker_path}")
        logger.info("=====================================================")

        if not PROCESSED_DATA_DIR.exists():
            logger.error(f"Processed data directory not found at {PROCESSED_DATA_DIR}")
            return

        tracker = IngestionTracker(self.tracker_path)
        
        txt_files = list(PROCESSED_DATA_DIR.glob("*.txt"))
        if not txt_files:
            logger.warning("No .txt files found to process.")
            tracker.close()
            return

        documents = []
        processed_file_paths = []

        for txt_file in txt_files:
            normalized_name = txt_file.stem + ".pdf"
            if tracker.is_already_processed(str(txt_file)):
                continue
                
            loader = TextLoader(str(txt_file), encoding="utf-8")
            loaded_docs = loader.load()
            
            for doc in loaded_docs:
                doc.metadata["source"] = normalized_name
                doc.metadata["access_group"] = "general"
                
            documents.extend(loaded_docs)
            processed_file_paths.append(txt_file)
            
        if not documents:
            logger.info("INCREMENTAL SYNC: No new documents. Exiting.")
            tracker.close()
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunked_documents = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunked_documents)} chunks from {len(processed_file_paths)} files.")

        # Batch Processing
        BATCH_SIZE = 50
        for i in range(0, len(chunked_documents), BATCH_SIZE):
            batch = chunked_documents[i : i + BATCH_SIZE]
            try:
                self.vectorstore.add_documents(documents=batch)
                logger.info(f"Successfully sent batch {(i // BATCH_SIZE) + 1} to Chroma server.")
            except Exception as e:
                logger.warning(f"Failed to send batch: {e}")
                
        if processed_file_paths:
            for file_path in processed_file_paths:
                tracker.mark_as_processed(str(file_path))

        tracker.close()
        logger.info("Finished indexing deal documents.")

    def delete_deal_document(self, filename: str):
        """
        Purges a document from the remote store and tracker.
        """
        tracker = IngestionTracker(self.tracker_path)
        try:
            matches = self.vectorstore.get(where={"source": filename})
            matching_ids = matches.get("ids", [])
            if matching_ids:
                self.vectorstore.delete(ids=matching_ids)
                tracker.remove_from_tracker(filename)
                return len(matching_ids)
            return 0
        finally:
            tracker.close()

def initialize_vector_store():
    """Compatibility wrapper for legacy calls."""
    store = ChromaDealStore()
    store.initialize_deal_store()

def delete_document_from_db(filename: str, chroma_dir: str = None):
    """Compatibility wrapper for legacy calls."""
    store = ChromaDealStore()
    return store.delete_deal_document(filename)

if __name__ == "__main__":
    initialize_vector_store()
