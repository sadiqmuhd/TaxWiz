"""RAG pipeline tests.

The unit tests below drive the real pipeline functions and only substitute the
two external services (OpenAI, Pinecone) at their client boundary, so prompt
construction, score filtering and error mapping are genuinely exercised.

The last test is an opt-in integration test: it talks to the real Pinecone index
and is skipped unless credentials are present.
"""

import os

import pytest

import rag


class FakeMatchSet(dict):
    """Mimics the mapping Pinecone's query() returns."""


@pytest.fixture(autouse=True)
def clear_client_cache():
    rag.reset_clients()
    yield
    rag.reset_clients()


def fake_index(matches):
    class Index:
        def __init__(self):
            self.last_query = None

        def query(self, **kwargs):
            self.last_query = kwargs
            return FakeMatchSet(matches=matches)

    return Index()


# ── Retrieval ──────────────────────────────────────────────────────────────────

def test_retrieve_returns_text_and_scores(monkeypatch):
    index = fake_index([
        {"id": "chunk-1", "score": 0.81, "metadata": {"text": "VAT is 7.5%."}},
        {"id": "chunk-2", "score": 0.44, "metadata": {"text": "Definition of goods."}},
    ])
    monkeypatch.setattr(rag, "get_pinecone_index", lambda: index)

    passages = rag.retrieve([0.1] * 1536)

    assert [p["id"] for p in passages] == ["chunk-1", "chunk-2"]
    assert passages[0]["text"] == "VAT is 7.5%."
    # The namespace must be passed, or the query silently searches an empty default.
    assert index.last_query["namespace"] == rag.PINECONE_NAMESPACE
    assert index.last_query["include_metadata"] is True


def test_retrieve_drops_weak_and_empty_matches(monkeypatch):
    index = fake_index([
        {"id": "good", "score": 0.60, "metadata": {"text": "Relevant passage."}},
        {"id": "weak", "score": 0.05, "metadata": {"text": "Unrelated passage."}},
        {"id": "blank", "score": 0.90, "metadata": {"text": "   "}},
        {"id": "nometa", "score": 0.90},
    ])
    monkeypatch.setattr(rag, "get_pinecone_index", lambda: index)

    assert [p["id"] for p in rag.retrieve([0.1] * 1536)] == ["good"]


def test_retrieve_handles_no_matches_at_all(monkeypatch):
    monkeypatch.setattr(rag, "get_pinecone_index", lambda: fake_index([]))
    assert rag.retrieve([0.1] * 1536) == []


def test_pinecone_failure_becomes_a_clean_rag_error(monkeypatch):
    class BrokenIndex:
        def query(self, **kwargs):
            raise ConnectionError("dns failure at 10.0.0.1")

    monkeypatch.setattr(rag, "get_pinecone_index", lambda: BrokenIndex())

    with pytest.raises(rag.RagError) as excinfo:
        rag.retrieve([0.1] * 1536)
    assert "knowledge base" in excinfo.value.message
    assert "10.0.0.1" not in excinfo.value.message  # no internals leaked


# ── Prompt construction ────────────────────────────────────────────────────────

def test_prompt_numbers_the_passages_for_citation():
    messages = rag.build_messages("What is VAT?", [
        {"id": "a", "score": 0.9, "text": "First passage."},
        {"id": "b", "score": 0.8, "text": "Second passage."},
    ])

    system, user = messages
    assert system["role"] == "system"
    assert rag.NO_ANSWER in system["content"]
    assert "[1] First passage." in user["content"]
    assert "[2] Second passage." in user["content"]
    assert "What is VAT?" in user["content"]


# ── End-to-end pipeline (external services faked at the client boundary) ───────

def test_empty_retrieval_says_so_instead_of_hallucinating(monkeypatch):
    monkeypatch.setattr(rag, "embed", lambda text: [0.0] * 1536)
    monkeypatch.setattr(rag, "retrieve", lambda vector, **kwargs: [])

    def must_not_be_called(messages):
        raise AssertionError("The LLM must not be called without context.")

    monkeypatch.setattr(rag, "generate", must_not_be_called)

    result = rag.answer_question("Who won the 2024 election?")

    assert result["answer"] == rag.NO_ANSWER
    assert result["grounded"] is False
    assert result["sources"] == []


def test_successful_answer_carries_its_sources(monkeypatch):
    monkeypatch.setattr(rag, "embed", lambda text: [0.0] * 1536)
    monkeypatch.setattr(rag, "retrieve", lambda vector, **kwargs: [
        {"id": "chunk-7", "score": 0.66, "text": "VAT is charged at 7.5 per cent."},
    ])
    monkeypatch.setattr(rag, "generate", lambda messages: "VAT is 7.5%. [1]")

    result = rag.answer_question("What is the VAT rate?")

    assert result["grounded"] is True
    assert result["answer"] == "VAT is 7.5%. [1]"
    assert result["sources"] == [
        {"id": "chunk-7", "score": 0.66, "excerpt": "VAT is charged at 7.5 per cent."}
    ]


@pytest.mark.parametrize("question", ["", "   ", None])
def test_blank_questions_are_rejected(question):
    with pytest.raises(ValueError):
        rag.answer_question(question)


def test_overlong_questions_are_rejected():
    with pytest.raises(ValueError):
        rag.answer_question("a" * (rag.MAX_QUESTION_CHARS + 1))


# ── Configuration ──────────────────────────────────────────────────────────────

def test_missing_api_key_raises_a_configuration_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(rag.ConfigurationError) as excinfo:
        rag.get_openai_client()
    assert "OPENAI_API_KEY" in str(excinfo.value)
    assert excinfo.value.status_code == 503


# ── Opt-in integration test ────────────────────────────────────────────────────

@pytest.mark.skipif(
    not os.getenv("PINECONE_API_KEY"),
    reason="PINECONE_API_KEY not set; skipping live Pinecone check",
)
def test_live_pinecone_index_is_reachable_and_populated():
    """Catches the failure this project actually had: an empty index of the
    wrong dimension. Requires real Pinecone credentials."""
    index = rag.get_pinecone_index()
    stats = index.describe_index_stats()

    assert stats["dimension"] == 1536, (
        "The index dimension must match text-embedding-3-small."
    )
    namespace = stats.get("namespaces", {}).get(rag.PINECONE_NAMESPACE)
    assert namespace is not None, (
        "Namespace '%s' does not exist in index '%s'. Run ingest.py."
        % (rag.PINECONE_NAMESPACE, rag.PINECONE_INDEX)
    )
    assert namespace["vector_count"] > 0, "The namespace holds no vectors."


# ── Ingestion chunking (pure, no network) ──────────────────────────────────────

def test_chunking_covers_the_document_without_looping_forever():
    import ingest

    text = " ".join("word%d" % i for i in range(3000))
    chunks = ingest.chunk(text)

    assert len(chunks) > 1
    assert all(0 < len(piece) <= ingest.CHUNK_CHARS for piece in chunks)
    assert chunks[0].startswith("word0")
    assert chunks[-1].endswith("word2999")


def test_chunking_handles_short_and_empty_documents():
    import ingest

    assert ingest.chunk("") == []
    assert ingest.chunk("   ") == []
    assert ingest.chunk("one two three") == ["one two three"]
    # Text with no whitespace still has to be split, not dropped.
    assert len(ingest.chunk("x" * 3000)) == 3
