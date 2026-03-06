import pytest
import sys
from pathlib import Path

# 1. Setup paths so pytest can find our src directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from ingestion.document_processor import PIIMasker

# 2. Pytest Fixture
# We use a fixture with scope="module" so the heavy SpaCy NLP model 
# only loads into memory ONCE for the entire test file, not before every single test.
@pytest.fixture(scope="module")
def masker():
    print("\nInitializing PIIMasker for tests...")
    return PIIMasker(model_name="en_core_web_lg")

# --- 3. The Test Cases ---

def test_mask_email_address(masker):
    """Verify that email addresses are fully redacted."""
    raw_text = "Please send the finalized term sheet to jane.smith@rbc.com immediately."
    masked_text = masker.mask_text(raw_text)
    
    assert "<EMAIL_ADDRESS>" in masked_text
    assert "jane.smith@rbc.com" not in masked_text

def test_mask_phone_number(masker):
    """Verify that phone numbers are fully redacted."""
    raw_text = "The lead syndication agent can be reached at 416-555-0198."
    masked_text = masker.mask_text(raw_text)
    
    assert "<PHONE_NUMBER>" in masked_text
    assert "416-555-0198" not in masked_text

def test_mask_person_name(masker):
    """Verify that individual human names are redacted."""
    raw_text = "The primary guarantor for this facility is Johnathan Doe."
    masked_text = masker.mask_text(raw_text)
    
    assert "<PERSON>" in masked_text
    assert "Johnathan Doe" not in masked_text

def test_retain_organization_names(masker):
    """
    CRITICAL TEST: Verify that corporate entities are NOT masked.
    If this fails, our RAG pipeline won't know which bank the deal belongs to.
    """
    raw_text = "This Credit Agreement is between Amerigo Resources Ltd. and Royal Bank of Canada."
    masked_text = masker.mask_text(raw_text)
    
    assert "Amerigo Resources Ltd." in masked_text
    assert "Royal Bank of Canada" in masked_text
    assert "<ORGANIZATION>" not in masked_text

def test_empty_string_handling(masker):
    """Verify the system doesn't crash on empty or whitespace-only documents."""
    assert masker.mask_text("") == ""
    assert masker.mask_text("   \n  ") == "   \n  "