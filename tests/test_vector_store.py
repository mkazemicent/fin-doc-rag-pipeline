import pytest
import sys
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

# 1. Setup paths so pytest can find our src directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def text_splitter():
    """
    Pre-configured splitter matching production settings in vector_store.py:
    chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""],
    )


@pytest.fixture(scope="module")
def long_financial_text():
    """
    ~2 000-char synthetic financial contract with realistic paragraph and
    line separators so the splitter can exercise its separator hierarchy.
    """
    return (
        "CREDIT AGREEMENT\n\n"
        "This Credit Agreement is entered into as of January 15, 2026, by and between "
        "Amerigo Resources Ltd. (the 'Borrower') and Royal Bank of Canada (the 'Lender').\n\n"
        "SECTION 1 - DEFINITIONS\n"
        "1.1 'Applicable Rate' means the annual rate of interest equal to the sum of the "
        "Reference Rate plus the Applicable Margin, calculated on the basis of a 365-day year.\n"
        "1.2 'Commitment' means the Lender's obligation to make Loans to the Borrower in an "
        "aggregate principal amount not to exceed CAD 150,000,000 at any time outstanding.\n\n"
        "SECTION 2 - THE FACILITY\n"
        "2.1 Subject to the terms and conditions set forth herein, the Lender agrees to make "
        "revolving credit loans to the Borrower from time to time during the Availability Period "
        "in an aggregate principal amount at any time outstanding not to exceed the Commitment.\n"
        "2.2 The Borrower may borrow, repay, and reborrow Loans during the Availability Period, "
        "provided that after giving effect to any such borrowing, the aggregate outstanding "
        "principal amount of all Loans does not exceed the Commitment.\n\n"
        "SECTION 3 - INTEREST AND FEES\n"
        "3.1 Each Loan shall bear interest on the outstanding principal amount thereof from the "
        "date of such Loan until maturity at a rate per annum equal to the Applicable Rate.\n"
        "3.2 The Borrower shall pay to the Lender a commitment fee equal to 0.25 percent per "
        "annum on the average daily unused portion of the Commitment during the Availability "
        "Period, payable quarterly in arrears.\n\n"
        "SECTION 4 - MATURITY DATE\n"
        "4.1 Unless earlier terminated, the Commitment shall expire and all outstanding Loans, "
        "together with accrued interest, shall be due and payable on the Maturity Date, which "
        "is defined as December 31, 2028.\n"
    )


def _make_fake_documents(n: int) -> list:
    """Create *n* LangChain Document objects with dummy page_content."""
    return [
        Document(page_content=f"Synthetic chunk number {i}.", metadata={"source": "test"})
        for i in range(n)
    ]


# ===========================================================================
# Category 1 — Chunk Size Limits  (RTX 3070 Ti / mxbai-embed-large guard)
# ===========================================================================

class TestChunkSizeLimits:
    """Every chunk produced by the production splitter must be ≤ 500 chars."""

    def test_no_chunk_exceeds_500_characters(self, text_splitter, long_financial_text):
        """
        CRITICAL: Feed a ~2 000-char realistic document into the splitter and
        assert that every resulting chunk respects the 500-character ceiling.
        """
        chunks = text_splitter.split_text(long_financial_text)

        assert len(chunks) > 1, "Text should produce multiple chunks"
        for idx, chunk in enumerate(chunks):
            assert len(chunk) <= 500, (
                f"Chunk {idx} is {len(chunk)} chars — exceeds the 500-char limit. "
                f"Preview: {chunk[:80]}..."
            )

    def test_single_short_document_produces_one_chunk(self, text_splitter):
        """A string well under 500 chars must not be fragmented."""
        short_text = "This is a short clause about the maturity date of the facility."
        chunks = text_splitter.split_text(short_text)

        assert len(chunks) == 1
        assert chunks[0] == short_text

    def test_chunk_size_with_no_natural_separators(self, text_splitter):
        """
        Adversarial edge case: a 1 500-char string with zero whitespace.
        The splitter must hard-cut at 500 chars when all separators are exhausted.
        """
        wall_of_text = "a" * 1500
        chunks = text_splitter.split_text(wall_of_text)

        assert len(chunks) > 1, "Continuous text should still be split"
        for idx, chunk in enumerate(chunks):
            assert len(chunk) <= 500, (
                f"Chunk {idx} is {len(chunk)} chars — hard-cut failed on separator-less input."
            )


# ===========================================================================
# Category 2 — Overlap Logic
# ===========================================================================

class TestOverlapLogic:
    """Validate that the 50-character overlap preserves semantic continuity."""

    @staticmethod
    def _find_overlap_length(chunk_a: str, chunk_b: str) -> int:
        """Return the length of the longest suffix of chunk_a that matches a prefix of chunk_b."""
        max_overlap = min(len(chunk_a), len(chunk_b))
        for length in range(max_overlap, 0, -1):
            if chunk_a.endswith(chunk_b[:length]):
                return length
        return 0

    def test_chunk_overlap_is_present(self, text_splitter):
        """
        For every pair of consecutive chunks, prove that the end of chunk[i]
        and the start of chunk[i+1] share a non-trivial overlapping region.

        NOTE: RecursiveCharacterTextSplitter aligns overlap to word boundaries,
        so we measure the actual suffix/prefix overlap rather than expecting an
        exact 50-character tail match.
        """
        # ~1 200 chars of continuous prose — no paragraph breaks
        continuous_text = (
            "The Borrower shall repay to the Lender the outstanding principal amount of each "
            "Loan on the applicable Maturity Date together with all accrued and unpaid interest "
            "thereon and all other amounts payable by the Borrower under this Credit Agreement. "
            "The Lender reserves the right to declare all outstanding Loans immediately due and "
            "payable upon the occurrence of any Event of Default as defined in Section Seven of "
            "this Credit Agreement including but not limited to failure to make any payment when "
            "due or the commencement of any bankruptcy insolvency or similar proceeding against "
            "the Borrower. The Applicable Rate shall be adjusted on the first Business Day of "
            "each calendar quarter based on the Reference Rate published by the Bank of Canada "
            "plus the Applicable Margin which shall be determined by the Pricing Grid set forth "
            "in Schedule B hereto. All payments shall be made in lawful money of Canada in "
            "immediately available funds to the account designated by the Lender. The Borrower "
            "hereby irrevocably waives any right to set off or counterclaim against any payment "
            "obligation arising under this Credit Agreement."
        )

        chunks = text_splitter.split_text(continuous_text)
        assert len(chunks) >= 2, "Need at least 2 chunks to verify overlap"

        for i in range(len(chunks) - 1):
            overlap = self._find_overlap_length(chunks[i], chunks[i + 1])
            assert overlap >= 10, (
                f"Insufficient overlap ({overlap} chars) between chunk {i} and {i + 1}.\n"
                f"  Chunk {i} tail:    {chunks[i][-60:]!r}\n"
                f"  Chunk {i+1} start: {chunks[i + 1][:60]!r}"
            )

    def test_overlap_does_not_exceed_chunk_size(self, text_splitter):
        """
        Configuration sanity check: overlap must always be strictly less than
        chunk_size. Violating this produces degenerate or erroring splits.
        """
        assert text_splitter._chunk_overlap < text_splitter._chunk_size, (
            f"Overlap ({text_splitter._chunk_overlap}) must be < "
            f"chunk_size ({text_splitter._chunk_size})"
        )


# ===========================================================================
# Category 3 — Batch Processing Resilience  (try/except in vector_store.py)
# ===========================================================================

class TestBatchProcessingResilience:
    """
    Verify the batch loop (lines 77-89 of vector_store.py) is fault-tolerant.
    We mock Chroma.add_documents so no GPU / Ollama / ChromaDB is needed.
    """

    @staticmethod
    def _run_batch_loop(chunked_documents, mock_vectorstore, batch_size=50):
        """
        Re-implementation of the production batch loop from vector_store.py
        so we can test its resilience logic in total isolation.
        """
        logger = logging.getLogger("test_batch_loop")
        successful_chunks = 0

        for i in range(0, len(chunked_documents), batch_size):
            batch = chunked_documents[i : i + batch_size]
            current_batch_num = (i // batch_size) + 1
            total_batches = (len(chunked_documents) // batch_size) + 1

            try:
                mock_vectorstore.add_documents(documents=batch)
                successful_chunks += len(batch)
                logger.info(f"Successfully embedded batch {current_batch_num}/{total_batches}")
            except Exception as e:
                logger.warning(
                    f"Failed to embed batch {current_batch_num}/{total_batches}. Error: {e}"
                )
                logger.warning("Skipping this batch to maintain pipeline stability.")
                continue

        return successful_chunks

    def test_batch_processing_continues_after_failure(self):
        """
        Simulate batch 1 failing (e.g. Ollama timeout) while batches 2 & 3
        succeed. The loop must NOT raise and must report 70 successful chunks.
        """
        docs = _make_fake_documents(120)  # 3 batches: 50 + 50 + 20

        mock_vs = MagicMock()
        mock_vs.add_documents.side_effect = [
            Exception("Simulated Ollama timeout"),  # batch 1 fails
            None,                                    # batch 2 succeeds
            None,                                    # batch 3 succeeds
        ]

        successful = self._run_batch_loop(docs, mock_vs)

        assert successful == 70, f"Expected 70 successful chunks, got {successful}"
        assert mock_vs.add_documents.call_count == 3

    def test_batch_processing_all_succeed(self):
        """Happy path: all batches succeed, chunk count must match total."""
        docs = _make_fake_documents(120)

        mock_vs = MagicMock()
        mock_vs.add_documents.return_value = None  # all calls succeed

        successful = self._run_batch_loop(docs, mock_vs)

        assert successful == 120
        assert mock_vs.add_documents.call_count == 3

    def test_batch_processing_all_fail(self):
        """
        Worst case: every batch fails (total GPU meltdown).
        The loop must still complete gracefully with 0 successful chunks.
        """
        docs = _make_fake_documents(120)

        mock_vs = MagicMock()
        mock_vs.add_documents.side_effect = Exception("GPU on fire")

        successful = self._run_batch_loop(docs, mock_vs)

        assert successful == 0
        assert mock_vs.add_documents.call_count == 3
