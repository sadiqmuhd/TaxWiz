# TaxWiz 🇳🇬⚖️

> A Nigerian tax assistant: deterministic tax calculations plus retrieval-augmented Q&A over a Nigerian tax corpus.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.1-lightgrey)
![Pinecone](https://img.shields.io/badge/Vector_DB-Pinecone-purple)
![Tests](https://img.shields.io/badge/tests-69%20passing-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

TaxWiz is a small Flask application with two halves that deliberately stay separate:

1. **A tax calculator.** PAYE, Company Income Tax, VAT and Withholding Tax are computed
   in plain Python (`tax.py`) from fixed rate tables. No model is involved in any
   arithmetic — the numbers are deterministic, unit-tested and reproducible.
2. **A tax Q&A assistant.** Questions are embedded, matched against a Pinecone index of
   Nigerian tax law passages, and answered by an LLM that is restricted to the retrieved
   text. When retrieval finds nothing relevant, TaxWiz says so instead of guessing, and
   every answer ships with the passages it was built from.

The split matters: language models are unreliable at arithmetic and reliable at reading.
TaxWiz uses each for what it is good at.

---

## Features

- **Deterministic tax engine** — PAYE (graduated bands, CRA/pension/NHF reliefs, 1%
  minimum tax), CIT (turnover-banded 0/20/30%), VAT (7.5%, inclusive or exclusive) and
  WHT (5% / 10% by transaction type).
- **Grounded Q&A** — retrieval-augmented answers with numbered citations and a source
  panel showing each passage and its similarity score.
- **Honest empty results** — below the similarity threshold, the LLM is never called;
  the user is told nothing relevant was found.
- **Clean JSON API** — every endpoint returns either data or `{"error", "message"}`.
  Stack traces never reach the user.
- **Graceful degradation** — missing or invalid credentials produce a readable message
  in the UI, and `/health` stays green so the platform does not kill the container.
- **Reproducible index** — `ingest.py` rebuilds the vector store from source documents.

---

## Architecture

```
                        Browser (templates/index.html)
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
         POST /api/calculate/<type>          POST /api/ask
                    │                               │
                    ▼                               ▼
              tax.py                              rag.py
      ┌─────────────────────┐        ┌──────────────────────────────┐
      │ fixed rate tables   │        │ 1. embed question (Gemini)   │
      │ pure functions      │        │ 2. search Pinecone namespace │
      │ no network, no LLM  │        │ 3. filter by similarity      │
      └─────────────────────┘        │ 4. build cited prompt        │
                    │                │ 5. generate answer (Gemini)  │
                    │                └──────────────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                        JSON response → rendered in the UI
```

`ingest.py` populates the Pinecone index offline and is not part of the request path.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | Flask 3.1 |
| Production server | Gunicorn (on Railway) |
| Vector database | Pinecone (serverless, cosine, 1536-dim) |
| Embeddings | Gemini `gemini-embedding-001` (1536-dim) |
| Generation | Gemini `gemini-3.6-flash` |
| Frontend | Server-rendered HTML, vanilla CSS + JavaScript (`fetch`) |
| Tests | pytest |

---

## Project Structure

```
TaxWiz/
├── app.py                  # Flask app: routes, validation, JSON error contract
├── tax.py                  # Deterministic tax calculations (no I/O, no LLM)
├── rag.py                  # Embedding → Pinecone retrieval → prompt → LLM
├── ingest.py               # Offline: chunk + embed documents into Pinecone
├── templates/index.html    # Single-page UI (chat + calculator)
├── tests/
│   ├── conftest.py
│   ├── test_tax.py         # Worked examples and input validation
│   ├── test_api.py         # HTTP routing, validation, error contract
│   └── test_rag.py         # Pipeline behaviour + live Pinecone check
├── requirements.txt        # Runtime dependencies
├── requirements-dev.txt    # + pytest
├── Procfile                # Railway start command
├── .python-version
└── .env.example
```

---

## Local Setup

**1. Clone and enter the project**

```bash
git clone https://github.com/sadiqmuhd/TaxWiz.git
```

```bash
cd TaxWiz
```

**2. Create a virtual environment**

```bash
python -m venv venv
```

Activate it — `venv\Scripts\activate` on Windows, `source venv/bin/activate` on macOS/Linux.

**3. Install dependencies**

```bash
pip install -r requirements-dev.txt
```

**4. Configure the environment**

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

You need a [Google AI Studio](https://aistudio.google.com/apikey) API key and a
[Pinecone](https://www.pinecone.io/) API key. `PINECONE_INDEX` and `PINECONE_NAMESPACE`
must point at a namespace built with the same embedding model — see below.

**5. Run**

```bash
python app.py
```

Open <http://127.0.0.1:5000>. The calculator works immediately; the chat assistant needs
both API keys and a populated index.

---

## Building the Knowledge Base

Put your source documents (`.txt`, `.md` or `.pdf`) in a `data/` folder, then:

```bash
python ingest.py --source data --create-index
```

This chunks each document (1200 characters, 200-character overlap), embeds the chunks
with `gemini-embedding-001`, and upserts them into `PINECONE_INDEX` under
`PINECONE_NAMESPACE`. Use `--dry-run` to preview the chunking without spending quota.

Because each vector keeps its own text in metadata, a namespace can also be re-embedded
with a different model without the original documents:

```bash
python ingest.py --from-namespace wiztax
```

That reads every passage out of the `wiztax` namespace and writes freshly embedded
copies into whatever `PINECONE_NAMESPACE` currently points at.

### One rule worth knowing

**Embeddings from different models are not comparable.** A `gemini-embedding-001` vector
and a `text-embedding-3-small` vector occupy unrelated spaces, so querying one model's
index with the other model's vector does not fail — it returns confidently wrong
passages. Keep one namespace per embedding model and switch between them with
`PINECONE_NAMESPACE`. This project uses `wiztax-gemini`; the older OpenAI-embedded
`wiztax` namespace is left intact alongside it.

Two details `ingest.py` and `rag.py` keep in sync, and which must not drift apart:

- **Task type.** Passages are embedded as `RETRIEVAL_DOCUMENT` and questions as
  `RETRIEVAL_QUERY`. Gemini treats the two asymmetrically; mixing them degrades results.
- **Dimension.** `gemini-embedding-001` returns 3072 dimensions by default and is
  truncated to `EMBEDDING_DIMENSIONS` (1536) to match the index, then renormalised to
  unit length — Gemini only normalises at full width.

## API

### `GET /health`

```json
{ "status": "healthy", "service": "TaxWiz",
  "gemini_configured": true, "pinecone_configured": true }
```

Never calls Gemini or Pinecone, so it stays green during an outage.

### `POST /api/calculate/<paye|cit|vat|wht>`

```bash
curl -X POST http://127.0.0.1:5000/api/calculate/paye \
  -H "Content-Type: application/json" \
  -d '{"gross_income": 1200000, "pension": 96000, "nhf": 30000}'
```

```json
{
  "tax_type": "paye",
  "total_relief": 566000.0,
  "taxable_income": 634000.0,
  "computed_tax": 59100.0,
  "minimum_tax": 12000.0,
  "minimum_tax_applied": false,
  "annual_tax": 59100.0,
  "monthly_tax": 4925.0,
  "effective_rate": 4.92,
  "bands": [{ "label": "First ₦300,000", "rate": 7.0, "tax": 21000.0, "applied": true }]
}
```

| Type | Required | Optional |
|---|---|---|
| `paye` | `gross_income` | `pension`, `nhf` |
| `cit` | `turnover`, `assessable_profit` | `capital_allowances` |
| `vat` | `amount` | `mode` (`exclusive` \| `inclusive`) |
| `wht` | `amount`, `transaction_type` | — |

### `POST /api/ask`

```bash
curl -X POST http://127.0.0.1:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the definition of banking?"}'
```

```json
{
  "answer": "Banking is business conducted or services offered by a bank. [1]",
  "sources": [{ "id": "chunk-124", "score": 0.62, "excerpt": "Definition of banking: ..." }],
  "grounded": true
}
```

When nothing relevant is retrieved, `grounded` is `false`, `sources` is empty, and the
answer says no relevant information was found.

### `GET /api/tax-types`

Lists supported calculators and the current VAT and WHT rates.

### Errors

Every failure returns the same shape:

```json
{ "error": "Invalid input", "message": "'gross_income' cannot be negative." }
```

`400` invalid input · `404` unknown endpoint · `405` wrong method ·
`502` upstream service failure · `503` missing configuration · `500` unexpected error.

---

## Railway Deployment

TaxWiz deploys as a standard Python service — no Docker, no extra infrastructure.

1. **Create the service.** In Railway, choose *New Project → Deploy from GitHub repo* and
   select this repository.
2. **Build.** Railway detects Python from `requirements.txt` and installs it. The Python
   version comes from `.python-version`.
3. **Start command.** Taken from the `Procfile`:

   ```
   web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
   ```

   Railway injects `PORT`; nothing is hardcoded.
4. **Environment variables.** Under *Variables*, set:

   | Variable | Required | Notes |
   |---|---|---|
   | `GEMINI_API_KEY` | yes | embeddings and generation |
   | `PINECONE_API_KEY` | yes | vector search |
   | `PINECONE_INDEX` | yes | e.g. `taxwiz2` |
   | `PINECONE_NAMESPACE` | yes | e.g. `wiztax-gemini` |
   | `GEMINI_CHAT_MODEL` | no | default `gemini-3.6-flash` |
   | `GEMINI_THINKING_LEVEL` | no | `low`, `high`, or `off` |
   | `GEMINI_EMBEDDING_MODEL` | no | default `gemini-embedding-001` |
   | `EMBEDDING_DIMENSIONS` | no | default `1536`, must match the index |
   | `RAG_TOP_K`, `RAG_MIN_SCORE` | no | retrieval tuning |

   Do **not** set `PORT` — Railway provides it.
5. **Health check.** Optionally set the healthcheck path to `/health` in the service
   settings. It responds without touching any external service.

---

## Testing

```bash
pytest
```

69 tests covering the tax engine (worked examples verified by hand, band boundaries,
input validation), the HTTP layer (routing, validation, the JSON error contract, no
stack-trace leakage) and the RAG pipeline (score filtering, prompt construction, empty
retrieval, upstream failure handling).

`tests/test_rag.py` also contains a live check that the configured Pinecone namespace
exists, has the right dimension and actually holds vectors. It is skipped automatically
when `PINECONE_API_KEY` is not set. If you point `PINECONE_NAMESPACE` at a namespace you
have not built yet, this test fails on purpose and tells you to run `ingest.py`.

---

## Limitations

- **The generation model is pinned one version back.** `gemini-3.6-flash`, not
  `gemini-3.7-flash`. The newer model returns `503 UNAVAILABLE` under free-tier load, and
  `gemini-flash-latest` resolves to it, so both proved unreliable. Switching is a
  one-line `GEMINI_CHAT_MODEL` change — `rag.py` adapts the thinking directive to the
  model generation and retries without it if the model rejects the newer form.

- **The corpus is small.** The index holds a few hundred passages, mostly statutory
  definitions. Questions outside that material correctly return "no relevant information
  was found" rather than an answer.
- **The PAYE schedule predates the 2025 reform.** The bands and reliefs implemented in
  `tax.py` are the graduated schedule this project was originally built against
  (7%–24%, CRA of ₦200,000 or 1% of gross plus 20%). They have been kept as-is and
  documented rather than silently changed. Verify against current Federal Inland Revenue
  Service guidance before relying on the figures.
- **Rates are hardcoded.** Changing a rate means editing `tax.py` and its tests.
- **No conversation memory.** Each question is answered independently.
- **Retrieval quality depends on ingestion.** Answers can only be as good as the
  documents indexed by `ingest.py`.
- **External dependencies.** Without valid Gemini and Pinecone credentials the chat
  assistant is unavailable; the calculator continues to work.

---

## Disclaimer

TaxWiz is an informational and technical demonstration project. It does not constitute
tax, legal or financial advice, and it does not replace a qualified tax professional.
Always verify figures and interpretations with the Federal Inland Revenue Service or a
certified practitioner before acting on them.

---

## License

MIT — see [LICENSE](LICENSE).
