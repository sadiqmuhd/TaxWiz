"""Retrieval-augmented question answering over the Nigerian tax corpus.

Flow: question -> embedding -> Pinecone similarity search -> context ->
prompt -> LLM -> answer. The model is instructed to answer only from the
retrieved context; when nothing relevant comes back we short-circuit and say so
instead of letting the model improvise.

Both the embeddings and the generated answer come from Google Gemini. The
embedding model must be the same one the index was built with — vectors from
different models are not comparable, and querying across them returns confident
nonsense rather than an error. See README "Building the Knowledge Base".

All configuration is read from the environment (see .env.example). Clients are
created lazily so the web app still starts, and /health still responds, when
credentials are missing.
"""

import logging
import math
import os
import threading
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# The SDK warns about automatic function calling on every generate_content call
# even when no tools are declared. TaxWiz declares none, so the warning is noise.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

# ── Configuration ──────────────────────────────────────────────────────────────

EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
# Deliberately not the newest model and not the floating alias. On the free
# tier gemini-3.7-flash returns 503 UNAVAILABLE under load, and
# gemini-flash-latest currently resolves to it, so both are unreliable;
# gemini-3.6-flash has served every request. Revisit when 3.7 has capacity.
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.6-flash")
# gemini-embedding-001 emits 3072 dimensions by default but supports truncation
# to 1536 or 768. 1536 is what the Pinecone index is provisioned for.
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

PINECONE_INDEX = os.getenv("PINECONE_INDEX", "taxwiz2")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "wiztax-gemini")

TOP_K = int(os.getenv("RAG_TOP_K", "5"))
# Cosine similarity below this is treated as "not really about the question".
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.25"))
MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "800"))
# "low"/"high" on Gemini 3.x, or "off" to send no thinking config at all.
THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "low")
MAX_QUESTION_CHARS = 1000

NO_ANSWER = (
    "No relevant information was found in the Nigerian tax corpus for that "
    "question. Try rephrasing it, or ask about a term the Act defines."
)

SYSTEM_PROMPT = """You are TaxWiz, an assistant for Nigerian tax law.

Answer ONLY from the numbered context passages provided in the user message.
If the context does not contain the answer, reply exactly:
"{no_answer}"

Do not use prior knowledge. Do not guess. Do not perform tax calculations —
direct the user to the TaxWiz calculator for figures. Keep answers concise and
cite the passage numbers you relied on, like [1] or [2]."""

USER_PROMPT = """Context passages:
{context}

Question: {question}"""


# ── Errors ─────────────────────────────────────────────────────────────────────

class RagError(Exception):
    """A failure the API can report to the user without leaking internals."""

    status_code = 502

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ConfigurationError(RagError):
    """A required API key or setting is missing."""

    status_code = 503


# ── Lazily created clients ─────────────────────────────────────────────────────

_lock = threading.Lock()
_gemini_client = None
_pinecone_index = None


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ConfigurationError(
            "%s is not configured. Set it in your .env file (local) or in the "
            "Railway service variables (deployed)." % name
        )
    return value


def get_gemini_client():
    global _gemini_client
    with _lock:
        if _gemini_client is None:
            from google import genai

            _gemini_client = genai.Client(api_key=_require("GEMINI_API_KEY"))
            logger.info("Gemini client initialised (chat=%s, embed=%s @ %dd)",
                        CHAT_MODEL, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS)
        return _gemini_client


def get_pinecone_index():
    global _pinecone_index
    with _lock:
        if _pinecone_index is None:
            from pinecone import Pinecone

            pc = Pinecone(api_key=_require("PINECONE_API_KEY"))
            _pinecone_index = pc.Index(PINECONE_INDEX)
            logger.info("Pinecone index '%s' connected (namespace='%s')",
                        PINECONE_INDEX, PINECONE_NAMESPACE)
        return _pinecone_index


def reset_clients() -> None:
    """Drop cached clients. Used by tests."""
    global _gemini_client, _pinecone_index
    with _lock:
        _gemini_client = None
        _pinecone_index = None


# ── Embeddings ─────────────────────────────────────────────────────────────────

def normalise(vector: List[float]) -> List[float]:
    """Scale a vector to unit length.

    Gemini's embeddings are only normalised at the full 3072 dimensions; once
    truncated they are not, so we renormalise to keep magnitudes consistent
    between what ingest.py stores and what a query produces.
    """
    magnitude = math.sqrt(sum(component * component for component in vector))
    if magnitude == 0:
        return vector
    return [component / magnitude for component in vector]


def embed_texts(texts: List[str], task_type: str) -> List[List[float]]:
    """Embed a batch of texts.

    `task_type` is RETRIEVAL_QUERY for questions and RETRIEVAL_DOCUMENT for
    indexed passages; Gemini embeds the two asymmetrically and mixing them up
    measurably degrades retrieval.
    """
    from google.genai import types

    client = get_gemini_client()
    try:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBEDDING_DIMENSIONS,
            ),
        )
    except Exception as exc:  # network, auth, rate limit
        logger.exception("Embedding request failed")
        raise RagError(
            "Could not reach the embedding service. Check GEMINI_API_KEY and try "
            "again. (%s)" % type(exc).__name__
        )
    return [normalise(list(item.values)) for item in response.embeddings]


def embed(text: str) -> List[float]:
    """Turn a user question into a query vector."""
    return embed_texts([text], task_type="RETRIEVAL_QUERY")[0]


# ── Retrieval ──────────────────────────────────────────────────────────────────

def retrieve(vector: List[float], top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """Fetch the most similar passages, keeping only confident matches."""
    index = get_pinecone_index()
    try:
        results = index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            namespace=PINECONE_NAMESPACE or None,
        )
    except Exception as exc:
        logger.exception("Pinecone query failed")
        raise RagError(
            "Could not reach the knowledge base. Check PINECONE_API_KEY and "
            "PINECONE_INDEX. (%s)" % type(exc).__name__
        )

    passages = []
    for match in results.get("matches", []) or []:
        metadata = match.get("metadata") or {}
        text = (metadata.get("text") or "").strip()
        score = float(match.get("score") or 0.0)
        if not text or score < MIN_SCORE:
            continue
        passages.append({"id": match.get("id"), "score": round(score, 4), "text": text})
    return passages


# ── Prompting and generation ───────────────────────────────────────────────────

def build_prompt(question: str, passages: List[Dict[str, Any]]) -> str:
    """Assemble the user message, numbering passages so they can be cited."""
    context = "\n\n".join(
        "[%d] %s" % (i, p["text"]) for i, p in enumerate(passages, start=1)
    )
    return USER_PROMPT.format(context=context, question=question)


def _generation_config(types, with_thinking: bool):
    """Build the request config, optionally including the thinking directive."""
    options = {
        "system_instruction": SYSTEM_PROMPT.format(no_answer=NO_ANSWER),
        "temperature": 0,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }
    # Gemini 3.x reasons before answering and bills that reasoning against
    # max_output_tokens; too small a budget is spent entirely on thinking and
    # returns an empty answer. TaxWiz only restates retrieved text, so keep
    # thinking minimal.
    if with_thinking and THINKING_LEVEL.lower() != "off":
        options["thinking_config"] = types.ThinkingConfig(thinking_level=THINKING_LEVEL)
    return types.GenerateContentConfig(**options)


def _rejected_the_argument(exc: Exception) -> bool:
    """True when the API refused the request shape rather than failing outright."""
    return getattr(exc, "code", None) == 400 or "INVALID_ARGUMENT" in str(exc)


def generate(prompt: str) -> str:
    """Ask the language model for an answer.

    `thinking_level` is a Gemini 3.x setting; the 2.5-era models take a numeric
    `thinking_budget` instead and reject it with a bare 400. Rather than tie the
    code to one model generation, drop the directive and retry once — so
    changing GEMINI_CHAT_MODEL really is a one-line change.
    """
    from google.genai import types

    client = get_gemini_client()
    response = None

    for with_thinking in (True, False):
        try:
            response = client.models.generate_content(
                model=CHAT_MODEL,
                contents=prompt,
                config=_generation_config(types, with_thinking),
            )
            break
        except Exception as exc:
            if with_thinking and _rejected_the_argument(exc):
                logger.info("%s rejected thinking_level=%s; retrying without it",
                            CHAT_MODEL, THINKING_LEVEL)
                continue
            logger.exception("Generation failed")
            raise RagError(
                "The language model is unavailable right now. Please try again. "
                "(%s)" % type(exc).__name__
            )

    answer = (response.text or "").strip()
    if not answer:
        logger.warning("Empty generation; finish reason: %s",
                       getattr(response.candidates[0], "finish_reason", "unknown")
                       if response.candidates else "no candidates")
        raise RagError("The language model returned an empty response.")
    return answer


def answer_question(question: str) -> Dict[str, Any]:
    """Run the full RAG pipeline and return the answer plus its sources."""
    question = (question or "").strip()
    if not question:
        raise ValueError("Question must not be empty.")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(
            "Question is too long (limit %d characters)." % MAX_QUESTION_CHARS
        )

    passages = retrieve(embed(question))
    if not passages:
        logger.info("No passages above score threshold for: %s", question)
        return {"answer": NO_ANSWER, "sources": [], "grounded": False}

    answer = generate(build_prompt(question, passages))

    # Passages can clear the similarity threshold and still not answer the
    # question, in which case the model returns the refusal text. Report that
    # as ungrounded so the response never claims support it does not have.
    if answer.strip().rstrip(".") == NO_ANSWER.rstrip("."):
        return {"answer": NO_ANSWER, "sources": [], "grounded": False}

    sources = [{"id": p["id"], "score": p["score"], "excerpt": p["text"][:220]}
               for p in passages]
    return {"answer": answer, "sources": sources, "grounded": True}
