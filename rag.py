"""Retrieval-augmented question answering over the Nigerian tax corpus.

Flow: question -> embedding -> Pinecone similarity search -> context ->
prompt -> LLM -> answer. The model is instructed to answer only from the
retrieved context; when nothing relevant comes back we short-circuit and say so
instead of letting the model improvise.

All configuration is read from the environment (see .env.example). Clients are
created lazily so the web app still starts, and /health still responds, when
credentials are missing.
"""

import logging
import os
import threading
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "taxwiz2")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "wiztax")
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
# Cosine similarity below this is treated as "not really about the question".
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.25"))
REQUEST_TIMEOUT = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
MAX_QUESTION_CHARS = 1000

NO_ANSWER = (
    "No relevant information was found in the Nigerian tax corpus for that "
    "question. Try rephrasing it, or ask about a term the Act defines."
)

SYSTEM_PROMPT = """You are TaxWiz, an assistant for Nigerian tax law.

Answer ONLY from the numbered context passages provided by the user message.
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
_openai_client = None
_pinecone_index = None


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ConfigurationError(
            "%s is not configured. Set it in your .env file (local) or in the "
            "Railway service variables (deployed)." % name
        )
    return value


def get_openai_client():
    global _openai_client
    with _lock:
        if _openai_client is None:
            from openai import OpenAI

            _openai_client = OpenAI(
                api_key=_require("OPENAI_API_KEY"), timeout=REQUEST_TIMEOUT
            )
            logger.info("OpenAI client initialised (chat=%s, embed=%s)",
                        CHAT_MODEL, EMBEDDING_MODEL)
        return _openai_client


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
    global _openai_client, _pinecone_index
    with _lock:
        _openai_client = None
        _pinecone_index = None


# ── Pipeline steps ─────────────────────────────────────────────────────────────

def embed(text: str) -> List[float]:
    """Turn a piece of text into a query vector."""
    client = get_openai_client()
    try:
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    except Exception as exc:  # network, auth, rate limit
        logger.exception("Embedding request failed")
        raise RagError(
            "Could not reach the embedding service. Check OPENAI_API_KEY and try "
            "again. (%s)" % type(exc).__name__
        )
    return response.data[0].embedding


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


def build_messages(question: str, passages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Assemble the chat messages, numbering passages so they can be cited."""
    context = "\n\n".join(
        "[%d] %s" % (i, p["text"]) for i, p in enumerate(passages, start=1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(no_answer=NO_ANSWER)},
        {"role": "user", "content": USER_PROMPT.format(context=context, question=question)},
    ]


def generate(messages: List[Dict[str, str]]) -> str:
    """Ask the language model for an answer."""
    client = get_openai_client()
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=400,
        )
    except Exception as exc:
        logger.exception("Chat completion failed")
        raise RagError(
            "The language model is unavailable right now. Please try again. "
            "(%s)" % type(exc).__name__
        )

    answer = (response.choices[0].message.content or "").strip()
    if not answer:
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

    answer = generate(build_messages(question, passages))
    sources = [{"id": p["id"], "score": p["score"], "excerpt": p["text"][:220]}
               for p in passages]
    return {"answer": answer, "sources": sources, "grounded": True}
