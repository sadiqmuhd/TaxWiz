"""RAG pipeline tests.

The unit tests below drive the real pipeline functions and only substitute the
two external services (Gemini, Pinecone) at their client boundary, so prompt
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
    prompt = rag.build_prompt("What is VAT?", [
        {"id": "a", "score": 0.9, "text": "First passage."},
        {"id": "b", "score": 0.8, "text": "Second passage."},
    ])

    assert "[1] First passage." in prompt
    assert "[2] Second passage." in prompt
    assert "What is VAT?" in prompt
    # The refusal wording lives in the system instruction, not the user turn.
    assert rag.NO_ANSWER in rag.SYSTEM_PROMPT.format(no_answer=rag.NO_ANSWER)


# ── End-to-end pipeline (external services faked at the client boundary) ───────

def test_empty_retrieval_says_so_instead_of_hallucinating(monkeypatch):
    monkeypatch.setattr(rag, "embed", lambda text: [0.0] * 1536)
    monkeypatch.setattr(rag, "retrieve", lambda vector, **kwargs: [])

    def must_not_be_called(prompt):
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
    monkeypatch.setattr(rag, "generate", lambda prompt: "VAT is 7.5%. [1]")

    result = rag.answer_question("What is the VAT rate?")

    assert result["grounded"] is True
    assert result["answer"] == "VAT is 7.5%. [1]"
    assert result["sources"] == [
        {"id": "chunk-7", "score": 0.66, "excerpt": "VAT is charged at 7.5 per cent."}
    ]


def test_a_refusal_is_reported_as_ungrounded(monkeypatch):
    """Passages can clear the score threshold without answering the question.
    When the model says so, the response must not claim sources."""
    monkeypatch.setattr(rag, "embed", lambda text: [0.0] * rag.EMBEDDING_DIMENSIONS)
    monkeypatch.setattr(rag, "retrieve", lambda vector, **kwargs: [
        {"id": "chunk-3", "score": 0.59, "text": "Definition of Minister: ..."},
    ])
    monkeypatch.setattr(rag, "generate", lambda prompt: rag.NO_ANSWER)

    result = rag.answer_question("Who won the 2019 election?")

    assert result["grounded"] is False
    assert result["sources"] == []
    assert result["answer"] == rag.NO_ANSWER


@pytest.mark.parametrize("question", ["", "   ", None])
def test_blank_questions_are_rejected(question):
    with pytest.raises(ValueError):
        rag.answer_question(question)


def test_overlong_questions_are_rejected():
    with pytest.raises(ValueError):
        rag.answer_question("a" * (rag.MAX_QUESTION_CHARS + 1))


# ── Generation config compatibility across model generations ───────────────────

class FakeGenerationClient:
    """Records each generate_content config and can reject the first shape."""

    def __init__(self, reject_thinking=False, error=None):
        self.reject_thinking = reject_thinking
        self.error = error
        self.configs = []
        self.models = self

    def generate_content(self, **kwargs):
        config = kwargs["config"]
        self.configs.append(config)
        if self.error is not None:
            raise self.error
        if self.reject_thinking and getattr(config, "thinking_config", None) is not None:
            raise RuntimeError("400 INVALID_ARGUMENT. Request contains an invalid argument.")

        class Response:
            text = "An answer. [1]"
            candidates = []

        return Response()


def test_thinking_level_is_sent_by_default(monkeypatch):
    client = FakeGenerationClient()
    monkeypatch.setattr(rag, "get_gemini_client", lambda: client)

    assert rag.generate("prompt") == "An answer. [1]"
    assert len(client.configs) == 1
    # The SDK normalises the string into a ThinkingLevel enum.
    level = client.configs[0].thinking_config.thinking_level
    assert rag.THINKING_LEVEL.upper() in str(level).upper()


def test_a_model_that_rejects_thinking_level_is_retried_without_it(monkeypatch):
    """Older Gemini models take a numeric thinking_budget and 400 on
    thinking_level. Switching GEMINI_CHAT_MODEL must not require a code change."""
    client = FakeGenerationClient(reject_thinking=True)
    monkeypatch.setattr(rag, "get_gemini_client", lambda: client)

    assert rag.generate("prompt") == "An answer. [1]"
    assert len(client.configs) == 2
    assert client.configs[0].thinking_config is not None
    assert client.configs[1].thinking_config is None


def test_a_real_outage_is_not_retried(monkeypatch):
    """A 503 means the model is down, not that the request shape is wrong —
    retrying would just burn a second request against the rate limit."""
    client = FakeGenerationClient(error=RuntimeError("503 UNAVAILABLE. high demand"))
    monkeypatch.setattr(rag, "get_gemini_client", lambda: client)

    with pytest.raises(rag.RagError):
        rag.generate("prompt")
    assert len(client.configs) == 1


def test_thinking_can_be_disabled_entirely(monkeypatch):
    client = FakeGenerationClient()
    monkeypatch.setattr(rag, "get_gemini_client", lambda: client)
    monkeypatch.setattr(rag, "THINKING_LEVEL", "off")

    rag.generate("prompt")
    assert client.configs[0].thinking_config is None


# ── Configuration ──────────────────────────────────────────────────────────────

def test_missing_api_key_raises_a_configuration_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(rag.ConfigurationError) as excinfo:
        rag.get_gemini_client()
    assert "GEMINI_API_KEY" in str(excinfo.value)
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

    assert stats["dimension"] == rag.EMBEDDING_DIMENSIONS, (
        "The index dimension must match the configured embedding dimension."
    )
    namespace = stats.get("namespaces", {}).get(rag.PINECONE_NAMESPACE)
    assert namespace is not None, (
        "Namespace '%s' does not exist in index '%s'. Run ingest.py."
        % (rag.PINECONE_NAMESPACE, rag.PINECONE_INDEX)
    )
    assert namespace["vector_count"] > 0, "The namespace holds no vectors."


# ── Embedding vectors ──────────────────────────────────────────────────────────

def test_vectors_are_normalised_to_unit_length():
    """Gemini only normalises at its full 3072 dimensions; truncated output is
    not normalised, so we do it ourselves and both sides must agree."""
    vector = rag.normalise([3.0, 4.0])
    assert vector == [0.6, 0.8]
    assert sum(component ** 2 for component in vector) == pytest.approx(1.0)


def test_normalising_a_zero_vector_does_not_divide_by_zero():
    assert rag.normalise([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


def test_queries_are_embedded_with_the_retrieval_query_task_type(monkeypatch):
    """Gemini embeds questions and documents asymmetrically. Using the document
    task type for a query silently degrades retrieval quality."""
    captured = {}

    class FakeModels:
        def embed_content(self, **kwargs):
            captured.update(kwargs)

            class Embedding:
                values = [1.0] + [0.0] * (rag.EMBEDDING_DIMENSIONS - 1)

            class Response:
                embeddings = [Embedding()]

            return Response()

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(rag, "get_gemini_client", lambda: FakeClient())

    vector = rag.embed("What is VAT?")

    assert captured["config"].task_type == "RETRIEVAL_QUERY"
    assert captured["config"].output_dimensionality == rag.EMBEDDING_DIMENSIONS
    assert captured["contents"] == ["What is VAT?"]
    assert len(vector) == rag.EMBEDDING_DIMENSIONS


def test_embedding_failure_becomes_a_clean_rag_error(monkeypatch):
    class FakeModels:
        def embed_content(self, **kwargs):
            raise RuntimeError("quota exceeded for project 12345")

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(rag, "get_gemini_client", lambda: FakeClient())

    with pytest.raises(rag.RagError) as excinfo:
        rag.embed("What is VAT?")
    assert "GEMINI_API_KEY" in excinfo.value.message
    assert "12345" not in excinfo.value.message


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
