import os
import logging
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

# Configure standard Python logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_chroma_instance(persist_directory: str):
    """
    Initializes a ChromaDB instance with the local mxbai-embed-large embedding model.
    """
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

def initialize_vector_store():
    """
    Reads processed .txt files, chunks them, generates fully local Ollama embeddings,
    and stores them in a local ChromaDB instance with robust batch processing.
    """
    logger.info("=====================================================")
    logger.info("IMPORTANT: Ensure your local Ollama application is running!")
    logger.info("=====================================================")

    # 1. Setup Paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
    CHROMA_DB_DIR = PROJECT_ROOT / "data" / "chroma_db"
    
    if not PROCESSED_DATA_DIR.exists():
        logger.error(f"Processed data directory not found at {PROCESSED_DATA_DIR}")
        return

    # 2. Initialize/Access ChromaDB Early (to check existing records)
    logger.info(f"Accessing ChromaDB at {CHROMA_DB_DIR}")
    vectorstore = get_chroma_instance(str(CHROMA_DB_DIR))
    
    # 2.1 Fetch Existing Sources
    try:
        # We only need the source metadata to determine what's already embedded
        existing_data = vectorstore.get(include=["metadatas"])
        existing_sources = {
            meta.get("source") 
            for meta in existing_data.get("metadatas", []) 
            if meta and "source" in meta
        }
        logger.info(f"Found {len(existing_sources)} document(s) already in vector store.")
    except Exception as e:
        logger.warning(f"Could not fetch existing metadata (DB might be empty): {e}")
        existing_sources = set()

    # 3. Load & Filter Documents
    logger.info(f"Loading and filtering .txt files from {PROCESSED_DATA_DIR}")
    txt_files = list(PROCESSED_DATA_DIR.glob("*.txt"))
    
    if not txt_files:
        logger.warning("No .txt files found to process.")
        return

    documents = []
    for txt_file in txt_files:
        normalized_name = txt_file.stem + ".pdf"
        
        # --- INCREMENTAL CHECK ---
        if normalized_name in existing_sources:
            continue  # Skip files already in DB
            
        loader = TextLoader(str(txt_file), encoding="utf-8")
        loaded_docs = loader.load()
        
        # Metadata Normalization
        for doc in loaded_docs:
            doc.metadata["source"] = normalized_name
            
        documents.extend(loaded_docs)
        
    if not documents:
        logger.info("=====================================================")
        logger.info("INCREMENTAL SYNC: No new documents to embed. Exiting.")
        logger.info("=====================================================")
        return

    logger.info(f"Identified {len(documents)} NEW document(s) to embed.")

    # 4. Chunking Strategy
    logger.info("Initializing RecursiveCharacterTextSplitter (Size: 500, Overlap: 50)")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunked_documents = text_splitter.split_documents(documents)
    logger.info(f"Total chunks created for new files: {len(chunked_documents)}")

    # 5. Initialize Local Ollama Embeddings
    logger.info("Initializing OllamaEmbeddings (Model: mxbai-embed-large)")
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    # 6. Robust Batch Processing
    BATCH_SIZE = 50
    total_batches = (len(chunked_documents) // BATCH_SIZE) + 1
    
    logger.info(f"Starting batch embedding process ({total_batches} total batches)...")
    
    successful_chunks = 0
    for i in range(0, len(chunked_documents), BATCH_SIZE):
        batch = chunked_documents[i : i + BATCH_SIZE]
        current_batch_num = (i // BATCH_SIZE) + 1
        
        try:
            vectorstore.add_documents(documents=batch)
            successful_chunks += len(batch)
            logger.info(f"Successfully embedded batch {current_batch_num}/{total_batches}")
        except Exception as e:
            logger.warning(f"Failed to embed batch {current_batch_num}/{total_batches}. Error: {e}")
            continue
            
    logger.info(f"Finished embedding. Stored {successful_chunks}/{len(chunked_documents)} chunks.")

    # 7. Quick Sanity Check
    test_query = "maturity date"
    logger.info(f"Running sanity check query: '{test_query}'")
    try:
        results = vectorstore.similarity_search(test_query, k=1)
        if results:
            preview = results[0].page_content[:150].replace('\n', ' ')
            source = results[0].metadata.get("source", "Unknown")
            logger.info(f"Sanity check passed. Top result from '{source}': {preview}...")
        else:
            logger.warning("Sanity check query returned no results.")
    except Exception as e:
         logger.error(f"Sanity check failed. Error: {e}")

def delete_document_from_db(filename: str, chroma_dir: str = None):
    """
    Purges all chunks associated with a specific document (source filename) from ChromaDB.

    Args:
        filename: Bare original PDF filename (e.g., 'amerigo_2015.pdf').
        chroma_dir: Optional path override for Chroma persistence.
    """
    if not chroma_dir:
        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
        chroma_dir = str(PROJECT_ROOT / "data" / "chroma_db")

    logger.info(f"Deleting all chunks for '{filename}' from ChromaDB at {chroma_dir}")
    
    vectorstore = get_chroma_instance(chroma_dir)
    
    # LangChain's Chroma.delete(where=...) has inconsistent support. 
    # Reliable pattern: get IDs -> delete by ID.
    try:
        matches = vectorstore.get(where={"source": filename})
        matching_ids = matches.get("ids", [])
        
        if matching_ids:
            vectorstore.delete(ids=matching_ids)
            logger.info(f"Successfully deleted {len(matching_ids)} chunks for '{filename}'.")
            return len(matching_ids)
        else:
            logger.warning(f"No chunks found in database for document: '{filename}'")
            return 0
            
    except Exception as e:
        logger.error(f"Failed to delete document '{filename}': {e}")
        return 0

if __name__ == "__main__":
    try:
        initialize_vector_store()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")