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


# ── Citation-based grounding ───────────────────────────────────────────────────

FIVE_PASSAGES = [
    {"id": "chunk-%d" % n, "score": 0.6, "text": "Passage %d text." % n}
    for n in range(1, 6)
]


def test_an_uncited_reply_is_withheld_not_shown_with_sources(monkeypatch):
    """Observed in production: "hey" produced a chatty greeting returned as
    grounded, with five unrelated passages attached as its "sources"."""
    monkeypatch.setattr(rag, "embed", lambda text: [0.0] * rag.EMBEDDING_DIMENSIONS)
    monkeypatch.setattr(rag, "retrieve", lambda vector, **kwargs: FIVE_PASSAGES)
    monkeypatch.setattr(rag, "generate", lambda prompt: (
        "Hello! I am TaxWiz, your assistant for Nigerian tax law. How can I help you today?"
    ))

    result = rag.answer_question("hey")

    assert result == {"answer": rag.NO_ANSWER, "sources": [], "grounded": False}


def test_an_answer_from_prior_knowledge_is_withheld(monkeypatch):
    """The real risk behind the greeting case: a confident tax claim the model
    did not take from any passage must never be presented as an answer."""
    monkeypatch.setattr(rag, "embed", lambda text: [0.0] * rag.EMBEDDING_DIMENSIONS)
    monkeypatch.setattr(rag, "retrieve", lambda vector, **kwargs: FIVE_PASSAGES)
    monkeypatch.setattr(rag, "generate", lambda prompt: "The top PAYE rate is 25%.")

    result = rag.answer_question("What is the top PAYE rate?")

    assert result["grounded"] is False
    assert "25%" not in result["answer"]


def test_only_the_passages_actually_cited_are_returned(monkeypatch):
    monkeypatch.setattr(rag, "embed", lambda text: [0.0] * rag.EMBEDDING_DIMENSIONS)
    monkeypatch.setattr(rag, "retrieve", lambda vector, **kwargs: FIVE_PASSAGES)
    monkeypatch.setattr(rag, "generate", lambda prompt: "Per the Act [2], and see [4].")

    result = rag.answer_question("question")

    assert result["grounded"] is True
    assert [s["id"] for s in result["sources"]] == ["chunk-2", "chunk-4"]


@pytest.mark.parametrize("answer,expected", [
    ("Banking is defined as ... [1].", [1]),
    ("See [1, 3].", [1, 3]),
    ("See [1][3].", [1, 3]),
    ("See [2-4].", [2, 3, 4]),
    ("See [3] and again [3].", [3]),
    ("No brackets at all.", []),
    ("A year [2025] is not a citation.", []),
    ("Invented [9] and [0] point at nothing.", []),
    ("A huge range [1-99999] is ignored.", []),
])
def test_citation_parsing(answer, expected):
    cited = rag.cited_passages(answer, FIVE_PASSAGES)
    assert [int(p["id"].split("-")[1]) for p in cited] == expected


@pytest.mark.parametrize("question", ["", "   ", None])
def test_blank_questions_are_rejected(question):
    with pytest.raises(ValueError):
        rag.answer_question(question)


def test_overlong_questions_are_rejected():
    with pytest.raises(ValueError):
        rag.answer_question("a" * (rag.MAX_QUESTION_CHARS + 1))


# ── Generation config compatibility across model generations ───────────────────

class FakeGenerationClient:
    """Records each generate_content config.

    `error` fails every call; `errors` fails the first len(errors) calls in
    turn and then succeeds; `reject_thinking` refuses any thinking directive.
    """

    def __init__(self, reject_thinking=False, error=None, errors=None):
        self.reject_thinking = reject_thinking
        self.error = error
        self.errors = list(errors or [])
        self.configs = []
        self.models = self

    def generate_content(self, **kwargs):
        config = kwargs["config"]
        self.configs.append(config)
        if self.error is not None:
            raise self.error
        if self.errors:
            raise self.errors.pop(0)
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


@pytest.fixture
def no_sleep(monkeypatch):
    """Skip the retry pause so tests stay fast; record what was requested."""
    pauses = []
    monkeypatch.setattr(rag.time, "sleep", pauses.append)
    return pauses


def test_a_transient_503_is_retried_once_and_recovers(monkeypatch, no_sleep):
    """Observed in production: Gemini returned 503 to a live user, and the same
    question succeeded seconds later. One quick retry absorbs that blip."""
    client = FakeGenerationClient(errors=[RuntimeError("503 UNAVAILABLE. high demand")])
    monkeypatch.setattr(rag, "get_gemini_client", lambda: client)

    assert rag.generate("prompt") == "An answer. [1]"
    assert len(client.configs) == 2
    assert no_sleep == [rag.TRANSIENT_RETRY_DELAY]


def test_a_persistent_outage_is_retried_once_then_reported(monkeypatch, no_sleep):
    """Retry exactly once: a model that is genuinely down must fail fast rather
    than hold the user's request open through a string of attempts."""
    client = FakeGenerationClient(error=RuntimeError("503 UNAVAILABLE. high demand"))
    monkeypatch.setattr(rag, "get_gemini_client", lambda: client)

    with pytest.raises(rag.RagError):
        rag.generate("prompt")
    assert len(client.configs) == 2


def test_a_quota_rejection_is_not_retried(monkeypatch, no_sleep):
    """A 429 will not clear in a second; retrying only spends more quota."""
    client = FakeGenerationClient(error=RuntimeError("429 RESOURCE_EXHAUSTED. quota"))
    monkeypatch.setattr(rag, "get_gemini_client", lambda: client)

    with pytest.raises(rag.RagError):
        rag.generate("prompt")
    assert len(client.configs) == 1
    assert no_sleep == []


def test_both_retries_together_stay_bounded(monkeypatch, no_sleep):
    """A shape rejection and a transient failure can both occur; the total is
    capped at three calls, never an open-ended loop."""
    client = FakeGenerationClient(errors=[
        RuntimeError("400 INVALID_ARGUMENT. bad thinking config"),
        RuntimeError("503 UNAVAILABLE."),
        RuntimeError("503 UNAVAILABLE."),
    ])
    monkeypatch.setattr(rag, "get_gemini_client", lambda: client)

    with pytest.raises(rag.RagError):
        rag.generate("prompt")
    assert len(client.configs) == 3


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


@pytest.mark.skipif(
    not os.getenv("PINECONE_API_KEY"),
    reason="PINECONE_API_KEY not set; skipping live corpus-quality check",
)
def test_live_corpus_contains_no_unreadable_passages():
    """The check above passed while 33 of 260 passages (12%) were
    character-reversed table dumps. Counting vectors is not checking content."""
    import ingest

    index = rag.get_pinecone_index()
    ids = [i for page in index.list(namespace=rag.PINECONE_NAMESPACE) for i in page]
    garbled = []
    for offset in range(0, len(ids), 100):
        fetched = index.fetch(ids=ids[offset:offset + 100], namespace=rag.PINECONE_NAMESPACE)
        for vector_id, vector in fetched.vectors.items():
            if ingest.looks_reversed((vector.metadata or {}).get("text") or ""):
                garbled.append(vector_id)

    assert garbled == [], (
        "Unreadable passages in '%s': %s. Run: python ingest.py --prune-garbled"
        % (rag.PINECONE_NAMESPACE, sorted(garbled))
    )


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


# ── Unreadable-passage detection (pure, no network) ────────────────────────────

# Verbatim text of passages that were in the live index before they were pruned.
# The originals survive in the 'wiztax' namespace.
REVERSED_SAMPLES = [
    # chunk-227 (first 400 characters), read back from the 'wiztax' namespace
    'Table: Table 2\nsnoitpmexE | ytreporp\n000,000,01N\not\nnaht\nsetaleR\nssel | Col3 | Col4 | ytreporp\n000,000,01N\not\nnaht\nsetaleR\nssel | Col6 | Col7 | Col8 | Col9\nelbail\nytuD\nsnosreP\nyaP\not | eegagtroM | eerefsnarT | eerefsnarT | eegagtroM | eerefsnarT | eeyaP | gnikat\nytiruces\nytraP | ynapmoC\netaR\nweN | %573.0 | %5.1 | %5.1 | %573.0 | %5.1 | %01.0 | %573.0 | %57.0\nepyT | merolaV\ndA | merolaV\ndA | merola',
    # chunk-234 (first 400 characters)
    'Table: Table 9\nCol1 | emas\nsâ€™eno\n-inortcele\nemas\n-ca\neht\notni\nro\ntnuocca neewteb eht nihtiw\ndiap\nderrefsnart\nfo\nyenoM stnuoc renwo\nnwo yllac knab\nelbail\ngniylrednu\n)noitcasnart\nsa\nemaS(\nrof | rorefsnarT\n05N | 05N\nytud\ndexiF | ytud\ndexiF\n)000,01N\nmorf\neulav\nrof(\ntpieceR | fo\nrefsnarT\nro\ntpiecer sdrawpu\ncinortcelE\n000,01N\n74 | 84\n',
]

READABLE_SAMPLES = [
    "Definition of banking: business conducted or services offered by a bank",
    "Definition of person: a company, partnership, community, family, individual, "
    "executor, trustee and legal arrangement",
    # Short and nearly free of function words: must not be mistaken for noise.
    "Definition of insurance: assurance",
]


@pytest.mark.parametrize("text", REVERSED_SAMPLES)
def test_reversed_extraction_is_detected(text):
    import ingest

    assert ingest.looks_reversed(text) is True


@pytest.mark.parametrize("text", READABLE_SAMPLES)
def test_readable_text_is_left_alone(text):
    import ingest

    assert ingest.looks_reversed(text) is False


def test_no_reversed_marker_word_is_itself_english():
    """If a marker were also a real word, normal prose could trip the detector."""
    import ingest

    assert ingest.FORWARD_WORDS.isdisjoint(ingest.REVERSED_WORDS)


def test_ingestion_drops_unreadable_passages_and_keeps_the_rest():
    import ingest

    pieces = [{"id": "good", "text": READABLE_SAMPLES[0], "source": "s"},
              {"id": "bad", "text": REVERSED_SAMPLES[0], "source": "s"}]

    assert [p["id"] for p in ingest.keep_readable(pieces)] == ["good"]


# ── Copying vectors between namespaces (fake index, no network) ────────────────

class FakeVector:
    def __init__(self, text):
        self.values = [0.1, 0.2]
        self.metadata = {"text": text}


class FakeNamespacedIndex:
    """Just enough of the Pinecone index API for copy_namespace."""

    def __init__(self, namespaces):
        self.namespaces = namespaces            # {name: {id: FakeVector}}
        self.upserts = []

    def list(self, namespace):
        yield list(self.namespaces.get(namespace, {}))

    def fetch(self, ids, namespace):
        class Fetched:
            vectors = {i: self.namespaces[namespace][i] for i in ids}
        return Fetched()

    def upsert(self, vectors, namespace):
        self.upserts.append((namespace, [v["id"] for v in vectors]))


def test_copy_namespace_copies_new_readable_vectors_only():
    import ingest

    index = FakeNamespacedIndex({
        "defs": {"chunk-1": FakeVector(READABLE_SAMPLES[0]),
                 "chunk-2": FakeVector(READABLE_SAMPLES[1]),
                 "chunk-3": FakeVector(REVERSED_SAMPLES[0])},
        "act": {"chunk-2": FakeVector(READABLE_SAMPLES[1])},     # already copied
    })

    copied = ingest.copy_namespace(index, "defs", "act", dry_run=False)

    assert copied == 1
    assert index.upserts == [("act", ["chunk-1"])]


def test_copy_namespace_dry_run_writes_nothing():
    import ingest

    index = FakeNamespacedIndex({"defs": {"chunk-1": FakeVector(READABLE_SAMPLES[0])}})

    assert ingest.copy_namespace(index, "defs", "act", dry_run=True) == 1
    assert index.upserts == []


def test_copy_namespace_refuses_to_copy_onto_itself():
    import ingest

    with pytest.raises(SystemExit):
        ingest.copy_namespace(FakeNamespacedIndex({}), "defs", "defs", dry_run=False)
