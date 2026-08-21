"""Build the TaxWiz knowledge base.

Two ways to populate the Pinecone namespace the app queries:

  # from source documents in a folder
  python ingest.py --source data

  # re-embed passages already stored in another namespace's metadata
  python ingest.py --from-namespace wiztax

The second form exists because the chunk text is kept in each vector's metadata,
so an index can be re-embedded with a different model without the original
documents. Embeddings from different models are not interchangeable — always
write them to a namespace of their own.
"""

import argparse
import glob
import logging
import os
import sys
import time
from typing import Dict, Iterator, List

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("ingest")

CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200
BATCH_SIZE = 50

# Gemini's free tier bills one request per *text*, not per batch, against a
# per-minute cap. Pace the upload to stay just under it rather than burning the
# quota and stalling for a minute at a time.
REQUESTS_PER_MINUTE = int(os.getenv("EMBED_REQUESTS_PER_MINUTE", "90"))
RETRY_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = 65


def read_documents(source_dir: str) -> Iterator[tuple]:
    """Yield (filename, text) for every supported document in `source_dir`."""
    patterns = ("*.txt", "*.md", "*.pdf")
    paths = sorted(
        path for pattern in patterns
        for path in glob.glob(os.path.join(source_dir, pattern))
    )
    if not paths:
        raise SystemExit(
            "No .txt, .md or .pdf files found in '%s'. Put your source documents "
            "there first." % source_dir
        )

    for path in paths:
        if path.lower().endswith(".pdf"):
            from pypdf import PdfReader

            text = "\n".join((page.extract_text() or "") for page in PdfReader(path).pages)
        else:
            with open(path, encoding="utf-8", errors="ignore") as handle:
                text = handle.read()
        yield os.path.basename(path), text


def chunk(text: str) -> List[str]:
    """Split text into overlapping windows, preferring to break at whitespace."""
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        if end < len(text):
            space = text.rfind(" ", start + CHUNK_OVERLAP, end)
            if space != -1:
                end = space
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def pieces_from_documents(source_dir: str) -> List[Dict[str, str]]:
    documents = list(read_documents(source_dir))
    pieces = []
    for name, text in documents:
        for position, body in enumerate(chunk(text)):
            pieces.append({"id": "%s-%d" % (os.path.splitext(name)[0], position),
                           "text": body, "source": name})
    logger.info("%d document(s) -> %d chunk(s)", len(documents), len(pieces))
    return pieces


def pieces_from_namespace(index, namespace: str) -> List[Dict[str, str]]:
    """Read every stored passage's text out of an existing namespace."""
    ids = [vector_id for page in index.list(namespace=namespace) for vector_id in page]
    if not ids:
        raise SystemExit("Namespace '%s' is empty — nothing to re-embed." % namespace)

    pieces = []
    for offset in range(0, len(ids), 100):
        fetched = index.fetch(ids=ids[offset:offset + 100], namespace=namespace)
        for vector_id, vector in fetched.vectors.items():
            metadata = vector.metadata or {}
            text = (metadata.get("text") or "").strip()
            if text:
                pieces.append({"id": vector_id, "text": text,
                               "source": metadata.get("source", namespace)})

    logger.info("read %d passage(s) from namespace '%s'", len(pieces), namespace)
    return pieces


def existing_ids(index, namespace: str) -> set:
    """Ids already written to the target namespace, so a resumed run can skip
    them instead of paying to embed the same passage twice."""
    try:
        return {vector_id for page in index.list(namespace=namespace)
                for vector_id in page}
    except Exception:
        return set()


def embed_with_retry(texts: List[str]) -> List[List[float]]:
    """Embed a batch, backing off when the rate limit rejects it."""
    import rag

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return rag.embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")
        except rag.RagError:
            if attempt == RETRY_ATTEMPTS:
                raise
            wait = RETRY_BACKOFF_SECONDS * attempt
            logger.warning("batch rejected (likely rate limit); retrying in %ds "
                           "[attempt %d/%d]", wait, attempt, RETRY_ATTEMPTS)
            time.sleep(wait)


def main() -> int:
    parser = argparse.ArgumentParser(description="Index passages into Pinecone for TaxWiz.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source", help="folder of source documents (.txt, .md, .pdf)")
    source.add_argument("--from-namespace",
                        help="re-embed passages already stored in this namespace")
    parser.add_argument("--create-index", action="store_true",
                        help="create the Pinecone index if it does not exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be written; do not embed or upsert")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-embed passages already present in the target namespace")
    args = parser.parse_args()

    if not args.source and not args.from_namespace:
        args.source = "data"

    # Import here so --help works without dependencies or credentials.
    import rag

    target = rag.PINECONE_NAMESPACE
    if args.from_namespace == target:
        raise SystemExit(
            "Refusing to re-embed namespace '%s' into itself. The target "
            "namespace (PINECONE_NAMESPACE) must differ from the source." % target
        )

    if args.source:
        pieces = pieces_from_documents(args.source)
        if args.dry_run:
            for piece in pieces[:3]:
                logger.info("sample %s: %s...", piece["id"], piece["text"][:120])
            logger.info("dry run: would write %d chunk(s) to namespace '%s'",
                        len(pieces), target)
            return 0

    for variable in ("GEMINI_API_KEY", "PINECONE_API_KEY"):
        if not os.getenv(variable):
            raise SystemExit("%s is not set. Copy .env.example to .env first." % variable)

    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    existing = [index.name for index in pc.list_indexes()]
    if rag.PINECONE_INDEX not in existing:
        if not args.create_index:
            raise SystemExit(
                "Pinecone index '%s' does not exist. Re-run with --create-index."
                % rag.PINECONE_INDEX
            )
        logger.info("Creating index '%s' (dim=%d, cosine)",
                    rag.PINECONE_INDEX, rag.EMBEDDING_DIMENSIONS)
        pc.create_index(
            name=rag.PINECONE_INDEX,
            dimension=rag.EMBEDDING_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    index = pc.Index(rag.PINECONE_INDEX)

    if args.from_namespace:
        pieces = pieces_from_namespace(index, args.from_namespace)
        if args.dry_run:
            logger.info("dry run: would re-embed %d passage(s) into '%s'",
                        len(pieces), target)
            return 0

    if not args.overwrite:
        already = existing_ids(index, target)
        skipped = [piece for piece in pieces if piece["id"] in already]
        pieces = [piece for piece in pieces if piece["id"] not in already]
        if skipped:
            logger.info("resuming: %d passage(s) already in '%s', %d to go",
                        len(skipped), target, len(pieces))
        if not pieces:
            logger.info("Nothing to do — namespace '%s' is already complete.", target)
            return 0

    logger.info("Embedding %d passage(s) with %s and writing to namespace '%s'",
                len(pieces), rag.EMBEDDING_MODEL, target)

    for offset in range(0, len(pieces), BATCH_SIZE):
        batch = pieces[offset:offset + BATCH_SIZE]
        vectors = [
            {
                "id": piece["id"],
                "values": embedding,
                "metadata": {"text": piece["text"], "source": piece["source"]},
            }
            for piece, embedding in zip(
                batch, embed_with_retry([piece["text"] for piece in batch])
            )
        ]
        index.upsert(vectors=vectors, namespace=target)
        done = min(offset + BATCH_SIZE, len(pieces))
        logger.info("upserted %d/%d", done, len(pieces))

        if done < len(pieces):
            pause = len(batch) * 60.0 / REQUESTS_PER_MINUTE
            logger.info("pausing %.0fs to stay under the embedding rate limit", pause)
            time.sleep(pause)

    logger.info("Done. Index '%s' namespace '%s': %s",
                rag.PINECONE_INDEX, target,
                index.describe_index_stats().get("namespaces", {}).get(target))
    return 0


if __name__ == "__main__":
    sys.exit(main())
