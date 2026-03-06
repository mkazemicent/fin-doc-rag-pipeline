import os
import glob
import logging
from pathlib import Path
from .hash_tracker import IngestionTracker
from langchain_community.document_loaders import PyPDFLoader
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

# Configure standard Python logging to output to the terminal at INFO level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PIIMasker:
    """
    A class that handles the detection and masking of Personally Identifiable Information (PII)
    in text using Microsoft Presidio and a SpaCy NLP model.
    """
    
    def __init__(self, model_name: str = "en_core_web_lg"):
        """
        Initializes the PIIMasker with the specified SpaCy model.
        
        Args:
            model_name (str): The name of the SpaCy model to use for the Analyzer.
        """
        logger.info(f"Initializing PIIMasker with SpaCy model: {model_name}")
        
        # Configure the Presidio NLP engine to use the specified SpaCy model
        base_configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": model_name}],
        }
        
        # Create the NLP engine using the provider
        provider = NlpEngineProvider(nlp_configuration=base_configuration)
        nlp_engine = provider.create_engine()
        
        # Initialize the Presidio AnalyzerEngine with our custom configured NLP engine
        self.analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine, 
            supported_languages=["en"]
        )
        
        # Initialize the Presidio AnonymizerEngine which performs the actual text replacement
        self.anonymizer = AnonymizerEngine()
        
        # Define the exact PII entities we want to detect and mask.
        # Note: 'ORGANIZATION' and 'LOCATION' are purposely excluded here to retain 
        # bank names and jurisdictions for the deal analysis.
        self.entities_to_mask = [
            "PERSON", 
            "EMAIL_ADDRESS", 
            "PHONE_NUMBER", 
            "IBAN_CODE", 
            "US_BANK_NUMBER"
        ]

    def mask_text(self, text: str) -> str:
        """
        Detects designated PII entities in the input text and masks them.
        
        Args:
            text (str): The raw text extracted from the document.
            
        Returns:
            str: The text with PII entities masked/anonymized.
        """
        # Return early if the text is empty or only contains whitespace
        if not text or not text.strip():
            return text
            
        # Analyze the text to find the specified entities
        analyzer_results = self.analyzer.analyze(
            text=text,
            entities=self.entities_to_mask,
            language="en"
        )
        
        # Anonymize (mask) the findings in the text based on the analysis results
        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=analyzer_results
        )
        
        return anonymized_result.text

def process_documents(raw_dir: str, processed_dir: str):
    """
    Iterates through all PDF files in the raw directory, extracts their text, 
    masks specific PII, and saves the output as .txt files in the processed directory.
    
    Args:
        raw_dir (str): Directory path containing raw PDF documents.
        processed_dir (str): Directory path where masked text files will be saved.
    """
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    
    # Ensure the output directory exists, creating parents if necessary
    processed_path.mkdir(parents=True, exist_ok=True)

    # Initialize the incremental ingestion tracker (SQLite + SHA-256)
    db_path = str(raw_path.parent / "ingestion_state.db")
    tracker = IngestionTracker(db_path)
    
    # Find all PDF files in the raw directory
    pdf_files = list(raw_path.glob("*.pdf"))
    if not pdf_files:
        logger.info(f"No PDF files found in {raw_dir}")
        return

    logger.info(f"Found {len(pdf_files)} PDF file(s) to process in {raw_dir}")
    
    # Instantiate the PIIMasker with the required en_core_web_lg SpaCy model
    masker = PIIMasker(model_name="en_core_web_lg")
    
    # Iterate through each PDF file to extract text and mask PII
    for pdf_file in pdf_files:
        logger.info(f"Processing started for file: {pdf_file.name}")
        
        # --- Incremental Ingestion Gate ---
        if tracker.is_already_processed(str(pdf_file)):
            logger.info(f"SKIPPED (already processed, unchanged): {pdf_file.name}")
            continue

        try:
            # Construct the output .txt file path
            output_file = processed_path / f"{pdf_file.stem}.txt"
            
            # Load the PDF using LangChain's PyPDFLoader
            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()
            
            # Extract and combine the text from all pages in the PDF
            full_text = "\n".join([page.page_content for page in pages])
            
            logger.info(f"Successfully extracted {len(pages)} page(s) from {pdf_file.name}. Applying PII masking...")
            
            # Mask the PII in the combined text
            masked_text = masker.mask_text(full_text)
            
            # Save the masked text to the processed directory
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(masked_text)
                
            logger.info(f"Successfully processed and saved masked content to: {output_file.name}")
            
            # Record this file as successfully processed
            tracker.mark_as_processed(str(pdf_file))
            
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {e}")

    # Close the tracker database connection
    tracker.close()

if __name__ == "__main__":
    # Resolve the project root path based on the current file's location 
    # Current location: src/ingestion/document_processor.py
    # Project root is 3 levels up
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    
    # Define absolute paths for the raw and processed data directories
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
    
    logger.info("=====================================================")
    logger.info("Starting Deal Analytics Document Processing Workflow")
    logger.info(f"Targeting raw directory: {RAW_DATA_DIR}")
    logger.info("=====================================================")
    
    process_documents(str(RAW_DATA_DIR), str(PROCESSED_DATA_DIR))
    
    logger.info("Workflow completed.")
