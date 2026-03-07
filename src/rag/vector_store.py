import os
import logging
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from src.ingestion.hash_tracker import IngestionTracker

# Load environment variables
load_dotenv('.env.local')

# Configure standard Python logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Module Level Path Constants ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
CHROMA_DB_DIR = PROJECT_ROOT / "data" / "chroma_db"
HASH_DB_PATH = PROJECT_ROOT / "data" / "ingestion_tracker.db"

def get_chroma_instance(persist_directory: str):
    """
    Initializes a ChromaDB instance with the local embedding model from environment.
    """
    embeddings = OllamaEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "mxbai-embed-large"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

def initialize_vector_store():
    """
    Reads processed .txt files, chunks them using a Semantic-Aware Recursive Strategy, 
    and stores them in a local ChromaDB instance.
    """
    logger.info("=====================================================")
    logger.info("IMPORTANT: Ensure your local Ollama application is running!")
    logger.info("=====================================================")

    if not PROCESSED_DATA_DIR.exists():
        logger.error(f"Processed data directory not found at {PROCESSED_DATA_DIR}")
        return

    # 2. Initialize Tracker and ChromaDB
    tracker = IngestionTracker(str(HASH_DB_PATH))
    logger.info(f"Accessing ChromaDB at {CHROMA_DB_DIR}")
    vectorstore = get_chroma_instance(str(CHROMA_DB_DIR))
    
    # 3. Load & Filter Documents based on SHA-256 Hashes
    logger.info(f"Loading and filtering .txt files from {PROCESSED_DATA_DIR}")
    txt_files = list(PROCESSED_DATA_DIR.glob("*.txt"))
    
    if not txt_files:
        logger.warning("No .txt files found to process.")
        tracker.close()
        return

    documents = []
    processed_file_paths = [] # To mark as processed later

    for txt_file in txt_files:
        normalized_name = txt_file.stem + ".pdf"
        
        # --- ROBUST INCREMENTAL CHECK (SHA-256) ---
        if tracker.is_already_processed(str(txt_file)):
            continue  # Skip files with identical content
            
        loader = TextLoader(str(txt_file), encoding="utf-8")
        loaded_docs = loader.load()
        
        # Metadata Normalization & RBAC Injection
        for doc in loaded_docs:
            doc.metadata["source"] = normalized_name
            doc.metadata["access_group"] = "general" # Phase 5: RBAC groundwork
            
        documents.extend(loaded_docs)
        processed_file_paths.append(txt_file)
        
    if not documents:
        logger.info("=====================================================")
        logger.info("INCREMENTAL SYNC: No new or modified documents. Exiting.")
        logger.info("=====================================================")
        tracker.close()
        return

    logger.info(f"Identified {len(documents)} NEW or MODIFIED document(s) for processing.")

    # 4. Semantic-Aware Recursive Strategy
    logger.info("Initializing RecursiveCharacterTextSplitter (Size: 800, Overlap: 150)")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunked_documents = text_splitter.split_documents(documents)
    logger.info(f"Total chunks created: {len(chunked_documents)}")

    # 5. Robust Batch Processing
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
            
    # 7. Update Tracker for successful files
    if successful_chunks > 0:
        for file_path in processed_file_paths:
            tracker.mark_as_processed(str(file_path))

    tracker.close()
    logger.info(f"Finished embedding. Stored {successful_chunks}/{len(chunked_documents)} chunks.")

    # 8. Quick Sanity Check
    test_query = "maturity date"
    logger.info(f"Running sanity check query: '{test_query}'")
    try:
        vectorstore = get_chroma_instance(str(CHROMA_DB_DIR))
        results = vectorstore.similarity_search(test_query, k=1)
        if results:
            res_doc = results[0]
            preview = res_doc.page_content[:150].replace('\n', ' ')
            source = res_doc.metadata.get("source", "Unknown")
            group = res_doc.metadata.get("access_group", "Unknown")
            
            # Detailed logging if metadata is missing
            if group == "Unknown":
                logger.warning(f"Metadata Warning: 'access_group' missing. Raw Metadata: {res_doc.metadata}")
            
            logger.info(f"Sanity check passed. Result from '{source}' (Group: {group}): {preview}...")
        else:
            logger.warning("Sanity check query returned no results.")
    except Exception as e:
         logger.error(f"Sanity check failed. Error: {e}")

def delete_document_from_db(filename: str, chroma_dir: str = None):
    """
    Purges all chunks associated with a specific document from ChromaDB 
    and removes its hash from the IngestionTracker.
    """
    if not chroma_dir:
        chroma_dir = str(CHROMA_DB_DIR)
    
    hash_db_path = str(HASH_DB_PATH)

    logger.info(f"Deleting all chunks for '{filename}' from ChromaDB at {chroma_dir}")
    
    vectorstore = get_chroma_instance(chroma_dir)
    tracker = IngestionTracker(hash_db_path)
    
    try:
        # 1. Delete from ChromaDB
        matches = vectorstore.get(where={"source": filename})
        matching_ids = matches.get("ids", [])
        
        if matching_ids:
            vectorstore.delete(ids=matching_ids)
            logger.info(f"Successfully deleted {len(matching_ids)} chunks from ChromaDB.")
            
            # 2. Delete from Tracker
            tracker.remove_from_tracker(filename)
            return len(matching_ids)
        else:
            logger.warning(f"No chunks found in database for document: '{filename}'")
            return 0
            
    except Exception as e:
        logger.error(f"Failed to delete document '{filename}': {e}")
        return 0
    finally:
        tracker.close()

if __name__ == "__main__":
    try:
        initialize_vector_store()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")