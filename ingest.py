"""Build the TaxWiz knowledge base.

Reads source documents (.txt, .md or .pdf) from a folder, splits them into
overlapping chunks, embeds each chunk with the same OpenAI model the app uses at
query time, and upserts the vectors into Pinecone.

    python ingest.py --source data --create-index

The embedding model and the index dimension must agree: text-embedding-3-small
produces 1536-dimension vectors, which is what `--create-index` provisions.
"""

import argparse
import glob
import logging
import os
import sys
from typing import Iterator, List

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("ingest")

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = 1536
INDEX_NAME = os.getenv("PINECONE_INDEX", "taxwiz2")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "wiztax")

CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200
BATCH_SIZE = 50


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Index documents into Pinecone for TaxWiz.")
    parser.add_argument("--source", default="data", help="folder of source documents")
    parser.add_argument("--create-index", action="store_true",
                        help="create the Pinecone index if it does not exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="chunk and report only; do not embed or upsert")
    args = parser.parse_args()

    documents = list(read_documents(args.source))
    pieces = []
    for name, text in documents:
        for position, body in enumerate(chunk(text)):
            pieces.append({"id": "%s-%d" % (os.path.splitext(name)[0], position),
                           "text": body, "source": name})
    logger.info("%d document(s) -> %d chunk(s)", len(documents), len(pieces))

    if args.dry_run:
        for piece in pieces[:3]:
            logger.info("sample %s: %s...", piece["id"], piece["text"][:120])
        return 0

    for variable in ("OPENAI_API_KEY", "PINECONE_API_KEY"):
        if not os.getenv(variable):
            raise SystemExit("%s is not set. Copy .env.example to .env first." % variable)

    from openai import OpenAI
    from pinecone import Pinecone, ServerlessSpec

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    existing = [index.name for index in pc.list_indexes()]
    if INDEX_NAME not in existing:
        if not args.create_index:
            raise SystemExit(
                "Pinecone index '%s' does not exist. Re-run with --create-index."
                % INDEX_NAME
            )
        logger.info("Creating index '%s' (dim=%d, cosine)", INDEX_NAME, EMBEDDING_DIMENSIONS)
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(INDEX_NAME)
    for offset in range(0, len(pieces), BATCH_SIZE):
        batch = pieces[offset:offset + BATCH_SIZE]
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL, input=[piece["text"] for piece in batch]
        )
        vectors = [
            {
                "id": piece["id"],
                "values": item.embedding,
                "metadata": {"text": piece["text"], "source": piece["source"]},
            }
            for piece, item in zip(batch, response.data)
        ]
        index.upsert(vectors=vectors, namespace=NAMESPACE)
        logger.info("upserted %d/%d", min(offset + BATCH_SIZE, len(pieces)), len(pieces))

    logger.info("Done. Index '%s' namespace '%s': %s",
                INDEX_NAME, NAMESPACE, index.describe_index_stats())
    return 0


if __name__ == "__main__":
    sys.exit(main())
