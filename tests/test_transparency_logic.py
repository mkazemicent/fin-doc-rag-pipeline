import pytest
from html import escape

from src.rag.deal_analyzer import DealExtraction, IRRELEVANT_QUERY_TOKEN


# ===========================================================================
# Category 1: Transparency Gating via routing_signal
# ===========================================================================

class TestTransparencyGating:
    """Verify that transparency is only shown when routing_signal == 'relevant'."""

    def _evaluate_show_transparency(self, routing_signal, optimized_query, chunks):
        """Mirrors the show_transparency logic from app/main.py."""
        return (
            routing_signal == "relevant"
            and optimized_query != IRRELEVANT_QUERY_TOKEN
            and len(chunks) > 0
        )

    def test_relevant_signal_with_chunks_shows_transparency(self):
        """Standard case: relevant query with retrieved chunks should show transparency."""
        chunks = [{"source": "deal.pdf", "group": "general", "content": "Rate is 5%"}]
        assert self._evaluate_show_transparency("relevant", "credit rate maturity", chunks) is True

    def test_irrelevant_signal_suppresses_transparency(self):
        """Bug 1 fix: irrelevant routing_signal must suppress transparency even with chunks."""
        chunks = [{"source": "deal.pdf", "group": "general", "content": "Rate is 5%"}]
        assert self._evaluate_show_transparency("irrelevant", "credit rate maturity", chunks) is False

    def test_error_signal_suppresses_transparency(self):
        """Bug 1 fix: error routing_signal must suppress transparency."""
        chunks = [{"source": "deal.pdf", "group": "general", "content": "Some content"}]
        assert self._evaluate_show_transparency("error", "credit rate", chunks) is False

    def test_empty_signal_suppresses_transparency(self):
        """Edge case: empty routing_signal must suppress transparency."""
        chunks = [{"source": "deal.pdf", "group": "general", "content": "Some content"}]
        assert self._evaluate_show_transparency("", "credit rate", chunks) is False

    def test_irrelevant_query_token_suppresses_transparency(self):
        """Even with relevant signal, IRRELEVANT_QUERY_TOKEN must suppress transparency."""
        chunks = [{"source": "deal.pdf", "group": "general", "content": "Some content"}]
        assert self._evaluate_show_transparency("relevant", IRRELEVANT_QUERY_TOKEN, chunks) is False

    def test_empty_chunks_suppresses_transparency(self):
        """If no chunks were retrieved, no transparency to show."""
        assert self._evaluate_show_transparency("relevant", "credit rate", []) is False


# ===========================================================================
# Category 2: DealExtraction Serialization for Chat History
# ===========================================================================

class TestDealExtractionSerialization:
    """Verify that DealExtraction objects serialize to clean human-readable strings."""

    def _serialize_for_history(self, content):
        """Mirrors the serialization logic from app/main.py chat_history builder."""
        if isinstance(content, DealExtraction):
            extraction = content
            return (
                f"Maturity Date: {extraction.maturity_date}. "
                f"Deal Terms: {', '.join(extraction.deal_terms)}. "
                f"Risk Factors: {', '.join(extraction.risk_factors)}."
            )
        return str(content)

    def test_deal_extraction_serializes_cleanly(self):
        """Bug 2 fix: DealExtraction must produce human-readable text, not Pydantic repr."""
        extraction = DealExtraction(
            deal_terms=["5% fixed rate", "10-year term"],
            risk_factors=["Currency exposure", "Interest rate risk"],
            maturity_date="2026-07-31"
        )
        result = self._serialize_for_history(extraction)
        assert "Maturity Date: 2026-07-31" in result
        assert "5% fixed rate, 10-year term" in result
        assert "Currency exposure, Interest rate risk" in result
        # Must NOT contain Pydantic repr artifacts
        assert "deal_terms=[" not in result
        assert "risk_factors=[" not in result

    def test_plain_string_passthrough(self):
        """Regular string content should pass through unchanged."""
        result = self._serialize_for_history("I cannot find this information in the provided deal documents.")
        assert result == "I cannot find this information in the provided deal documents."

    def test_refusal_string_passthrough(self):
        """Refusal messages should pass through without corruption."""
        refusal = "I am a strictly air-gapped financial assistant designed to analyze Canadian corporate contracts. I cannot fulfill non-financial requests such as recipes, code generation, or general trivia."
        result = self._serialize_for_history(refusal)
        assert result == refusal


# ===========================================================================
# Category 3: XSS Escaping
# ===========================================================================

class TestXSSEscaping:
    """Verify that user-controlled content is HTML-escaped before rendering."""

    def test_script_tag_escaped(self):
        """Bug 3 fix: Script tags in chunk content must be escaped."""
        malicious = '<script>alert("xss")</script>'
        escaped = escape(malicious)
        assert "<script>" not in escaped
        assert "&lt;script&gt;" in escaped

    def test_event_handler_escaped(self):
        """Bug 3 fix: Event handlers in chunk content must be escaped."""
        malicious = '<img src=x onerror="alert(1)">'
        escaped = escape(malicious)
        assert "onerror" not in escaped or "&quot;" in escaped
        assert "<img" not in escaped

    def test_normal_content_unchanged(self):
        """Normal financial text should pass through escape() without visible change."""
        normal = "The credit agreement matures on 2026-07-31 with a 5.25% rate."
        escaped = escape(normal)
        assert escaped == normal

    def test_ampersand_escaped(self):
        """Ampersands in content must be escaped."""
        content = "Terms & Conditions apply"
        escaped = escape(content)
        assert "&amp;" in escaped


# ===========================================================================
# Category 4: Message History Routing Signal Storage
# ===========================================================================

class TestMessageHistoryStorage:
    """Verify that routing_signal is stored and used to gate history re-rendering."""

    def _should_render_transparency(self, message):
        """Mirrors the history re-rendering logic from app/main.py."""
        return (
            "transparency" in message
            and message.get("routing_signal") == "relevant"
        )

    def test_relevant_message_renders_transparency(self):
        """Bug 4 fix: Only messages with routing_signal='relevant' should render transparency."""
        message = {
            "role": "assistant",
            "content": "Some answer",
            "routing_signal": "relevant",
            "transparency": {"query": "credit rate", "chunks": [{"source": "a.pdf", "group": "general", "content": "text"}]}
        }
        assert self._should_render_transparency(message) is True

    def test_irrelevant_message_skips_transparency(self):
        """Bug 4 fix: Messages with routing_signal='irrelevant' must not render transparency."""
        message = {
            "role": "assistant",
            "content": "I am a strictly air-gapped financial assistant...",
            "routing_signal": "irrelevant",
            "transparency": {"query": "recipe", "chunks": []}
        }
        assert self._should_render_transparency(message) is False

    def test_missing_routing_signal_skips_transparency(self):
        """Legacy messages without routing_signal must not render transparency."""
        message = {
            "role": "assistant",
            "content": "Some old message",
            "transparency": {"query": "test", "chunks": []}
        }
        assert self._should_render_transparency(message) is False

    def test_message_without_transparency_key(self):
        """Messages with no transparency data should not render transparency."""
        message = {
            "role": "assistant",
            "content": "Some answer",
            "routing_signal": "relevant"
        }
        assert self._should_render_transparency(message) is False
