import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from .hash_tracker import IngestionTracker
from langchain_community.document_loaders import PyPDFLoader
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from src.config import Settings, get_settings

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

def _process_single_file(
    pdf_file: Path, processed_path: Path, masker: "PIIMasker"
) -> Path:
    """CPU-bound worker: load PDF → mask PII → write .txt.

    Returns output_file path on success.
    Raises on failure so the caller can log and continue.
    """
    output_file = processed_path / f"{pdf_file.stem}.txt"

    loader = PyPDFLoader(str(pdf_file))
    pages = loader.load()
    full_text = "\n".join([page.page_content for page in pages])

    logger.info(
        f"Successfully extracted {len(pages)} page(s) from {pdf_file.name}. Applying PII masking..."
    )

    masked_text = masker.mask_text(full_text)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(masked_text)

    logger.info(f"Successfully processed and saved masked content to: {output_file.name}")
    return output_file


def process_documents(raw_dir: str, processed_dir: str, settings: Optional[Settings] = None, masker: Optional["PIIMasker"] = None, progress_callback=None) -> None:
    """
    Iterates through all PDF files in the raw directory, extracts their text,
    masks specific PII, and saves the output as .txt files in the processed directory.

    CPU-bound steps (load, mask, chunk) run in parallel via ThreadPoolExecutor.
    The hash-tracker gate and mark-as-processed steps remain sequential.

    Args:
        raw_dir (str): Directory path containing raw PDF documents.
        processed_dir (str): Directory path where masked text files will be saved.
        settings: Optional Settings instance; defaults to global settings.
        masker: Optional PIIMasker instance; created if not provided.
        progress_callback: Optional callable(completed, total) for UI progress.
    """
    settings = settings or get_settings()

    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)

    # Ensure the output directory exists
    processed_path.mkdir(parents=True, exist_ok=True)

    db_path = str(settings.hash_db_path)

    # Find all PDF files in the raw directory
    pdf_files = list(raw_path.glob("*.pdf"))
    if not pdf_files:
        logger.info(f"No PDF files found in {raw_dir}")
        return

    logger.info(f"Found {len(pdf_files)} PDF file(s) to process in {raw_dir}")

    # Instantiate the PIIMasker with the required en_core_web_lg SpaCy model
    if masker is None:
        masker = PIIMasker(model_name="en_core_web_lg")

    with IngestionTracker(db_path) as tracker:
        # --- Sequential gate: determine which files need processing ---
        files_to_process: list[tuple[Path, str]] = []
        for pdf_file in pdf_files:
            already_processed, file_hash = tracker.check_and_hash(str(pdf_file))
            if already_processed:
                logger.info(f"SKIPPED (already processed, unchanged): {pdf_file.name}")
                continue
            files_to_process.append((pdf_file, file_hash))

        if not files_to_process:
            logger.info("All files already processed.")
            return

        total = len(files_to_process)
        completed = 0

        # --- Fan-out CPU-bound work across threads ---
        future_to_info: dict = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            for pdf_file, file_hash in files_to_process:
                logger.info(f"Submitting for processing: {pdf_file.name}")
                future = executor.submit(
                    _process_single_file, pdf_file, processed_path, masker
                )
                future_to_info[future] = (pdf_file, file_hash)

            for future in as_completed(future_to_info):
                pdf_file, file_hash = future_to_info[future]
                try:
                    future.result()
                    tracker.mark_as_processed_with_hash(str(pdf_file), file_hash)
                except Exception as e:
                    logger.error(f"Error processing {pdf_file.name}: {e}")
                finally:
                    completed += 1
                    if progress_callback is not None:
                        progress_callback(completed, total)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    settings = get_settings()
    RAW_DATA_DIR = settings.data_root / "raw"
    PROCESSED_DATA_DIR = settings.processed_data_dir

    logger.info("=====================================================")
    logger.info("Starting Deal Analytics Document Processing Workflow")
    logger.info(f"Targeting raw directory: {RAW_DATA_DIR}")
    logger.info("=====================================================")

    process_documents(str(RAW_DATA_DIR), str(PROCESSED_DATA_DIR), settings=settings)

    logger.info("Workflow completed.")
