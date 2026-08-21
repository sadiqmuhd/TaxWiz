"""TaxWiz — Flask web app.

Two things live behind this API:
  * a deterministic tax calculator (tax.py), and
  * a retrieval-augmented Q&A pipeline over Nigerian tax law (rag.py).

Run locally with `python app.py`; in production Railway starts it with
gunicorn (see Procfile).
"""

import logging
import os
import sys

from flask import Flask, jsonify, render_template, request

import rag
import tax

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("taxwiz")

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


# ── Pages ──────────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    """Liveness probe. Deliberately does not touch OpenAI or Pinecone."""
    return jsonify({
        "status": "healthy",
        "service": "TaxWiz",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "pinecone_configured": bool(os.getenv("PINECONE_API_KEY")),
    })


# ── API ────────────────────────────────────────────────────────────────────────

@app.post("/api/ask")
def ask():
    """Answer a Nigerian tax law question from the indexed corpus."""
    payload = request.get_json(silent=True) or {}
    question = payload.get("question")

    if not isinstance(question, str) or not question.strip():
        return error("Invalid request", "Field 'question' is required.", 400)

    try:
        result = rag.answer_question(question)
    except ValueError as exc:
        return error("Invalid request", str(exc), 400)
    except rag.RagError as exc:
        logger.warning("RAG failure: %s", exc.message)
        return error("Service unavailable", exc.message, exc.status_code)
    except Exception:
        logger.exception("Unexpected failure answering question")
        return error("Internal error", "Something went wrong answering that question.", 500)

    return jsonify(result)


@app.post("/api/calculate/<tax_type>")
def calculate(tax_type):
    """Compute PAYE, CIT, VAT or WHT from user-supplied figures."""
    payload = request.get_json(silent=True)
    if payload is None:
        return error("Invalid request", "A JSON request body is required.", 400)

    try:
        result = tax.calculate(tax_type, payload)
    except tax.TaxInputError as exc:
        return error("Invalid input", str(exc), 400)
    except Exception:
        logger.exception("Unexpected failure in tax calculation")
        return error("Internal error", "Could not complete that calculation.", 500)

    return jsonify(result)


@app.get("/api/tax-types")
def tax_types():
    """Describe what the calculator supports (used to populate the WHT list)."""
    return jsonify({
        "tax_types": sorted(tax.CALCULATORS),
        "vat_rate": tax.VAT_RATE * 100,
        "wht_rates": {
            key: {"rate": value["rate"] * 100, "label": value["label"]}
            for key, value in tax.WHT_RATES.items()
        },
    })


# ── Error handling ─────────────────────────────────────────────────────────────

def error(title, message, status):
    """Every failure leaves the app as the same small JSON shape."""
    return jsonify({"error": title, "message": message}), status


@app.errorhandler(404)
def not_found(_exc):
    if request.path.startswith("/api/"):
        return error("Not found", "No such endpoint: %s" % request.path, 404)
    return render_template("index.html"), 404


@app.errorhandler(405)
def method_not_allowed(_exc):
    return error("Method not allowed",
                 "%s is not allowed on %s." % (request.method, request.path), 405)


@app.errorhandler(500)
def server_error(_exc):
    return error("Internal error", "An unexpected error occurred.", 500)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_DEBUG") == "1")
