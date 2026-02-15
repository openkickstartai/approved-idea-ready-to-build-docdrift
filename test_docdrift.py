"""Comprehensive tests for DocDrift — unit, integration, and property-based."""
import pytest
from hypothesis import given, strategies as st
from docdrift import (
    tokenize, split_identifier, cosine_similarity,
    extract_code_elements, extract_doc_sections, compute_drift, analyze,
)


class TestTokenize:
    def test_basic_words(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_with_numbers(self):
        assert tokenize("item2 val3") == ["item2", "val3"]

    def test_empty_string(self):
        assert tokenize("") == []

    def test_pure_digits_ignored(self):
        assert tokenize("123 456") == []


class TestSplitIdentifier:
    def test_snake_case(self):
        assert split_identifier("my_function") == ["my", "function"]

    def test_camel_case(self):
        assert split_identifier("myFunction") == ["my", "function"]

    def test_single_lowercase(self):
        assert split_identifier("run") == ["run"]


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert cosine_similarity(["a", "b"], ["a", "b"]) == pytest.approx(1.0)

    def test_completely_disjoint(self):
        assert cosine_similarity(["a", "b"], ["x", "y"]) == pytest.approx(0.0)

    def test_empty_input(self):
        assert cosine_similarity([], ["a"]) == 0.0
        assert cosine_similarity(["a"], []) == 0.0

    def test_partial_overlap(self):
        sim = cosine_similarity(["a", "b", "c"], ["a", "b", "z"])
        assert 0.0 < sim < 1.0


class TestExtractCodeElements:
    def test_function_with_docstring(self):
        code = 'def add(x, y):\n    """Add two numbers."""\n    return x + y\n'
        elems = extract_code_elements(code)
        assert len(elems) == 1
        assert elems[0]["name"] == "add"
        assert elems[0]["type"] == "function"
        assert elems[0]["docstring"] == "Add two numbers."

    def test_function_no_docstring(self):
        elems = extract_code_elements("def foo():\n    return 1\n")
        assert elems[0]["docstring"] == ""

    def test_class_detected(self):
        code = 'class Calc:\n    """A calculator."""\n    x = 1\n'
        elems = extract_code_elements(code)
        names = [e["name"] for e in elems]
        assert "Calc" in names
        assert any(e["type"] == "class" for e in elems)


class TestExtractDocSections:
    def test_two_sections(self):
        md = "# Title\nContent one\n## Sub\nContent two"
        secs = extract_doc_sections(md)
        assert len(secs) == 2
        assert secs[0]["heading"] == "Title"
        assert secs[1]["heading"] == "Sub"

    def test_no_headings(self):
        assert extract_doc_sections("just plain text") == []


class TestAnalyzeIntegration:
    def test_aligned_docstring(self):
        code = 'def calculate_sum(numbers):\n    """Calculate the sum of numbers."""\n    total = sum(numbers)\n    return total\n'
        reports = analyze(code, threshold=0.8)
        assert reports[0]["status"] == "aligned"
        assert reports[0]["drift"] < 0.8

    def test_missing_docstring_flagged(self):
        reports = analyze("def mystery():\n    return 42\n")
        assert reports[0]["status"] == "missing_docs"
        assert reports[0]["drift"] == 1.0

    def test_drifted_docstring_detected(self):
        code = (
            'def send_email(recipient, body):\n'
            '    """Parse XML config and validate schema."""\n'
            '    server = connect(recipient)\n'
            '    server.deliver(body)\n'
        )
        reports = analyze(code, threshold=0.5)
        assert reports[0]["status"] == "drifted"
        assert reports[0]["drift"] >= 0.5


class TestPropertyBased:
    @given(st.text(alphabet=st.characters(whitelist_categories=("Ll",)), min_size=2, max_size=30))
    def test_self_similarity_is_one(self, word):
        tokens = tokenize(word)
        if tokens:
            assert cosine_similarity(tokens, tokens) == pytest.approx(1.0)

    @given(st.text(min_size=0, max_size=80))
    def test_drift_always_bounded(self, text):
        score = compute_drift(text, ["code", "token", "example"])
        assert 0.0 <= score <= 1.0

    @given(st.text(alphabet=st.characters(whitelist_categories=("Ll",)), min_size=1, max_size=20))
    def test_tokenize_always_lowercase(self, text):
        for token in tokenize(text):
            assert token == token.lower()
            assert len(token) >= 1
