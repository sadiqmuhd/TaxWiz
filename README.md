# TaxWiz 🇳🇬⚖️

> A Nigerian tax assistant: deterministic tax calculations plus retrieval-augmented Q&A over a Nigerian tax corpus.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.1-lightgrey)
![Pinecone](https://img.shields.io/badge/Vector_DB-Pinecone-purple)
![Tests](https://img.shields.io/badge/tests-210%20passing-brightgreen)
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

Both halves are reachable from one chat box. A rule-based router (`chat.py`) decides
whether each message is asking for a figure or for information. Figures go to the
calculator and questions go to the Q&A pipeline. The router is not a language model, so
whether to calculate, and with what numbers, is never decided by AI.

---

## Features

- **One conversational assistant** — "I earn ₦500,000 monthly. Calculate my PAYE."
  returns a calculated breakdown; "What is PAYE?" returns a sourced explanation. When a
  calculation is missing a figure, TaxWiz asks for it rather than assuming one.
- **Deterministic tax engine, written from the Nigeria Tax Act 2025** — PAYE (Fourth
  Schedule bands 0–25%; s. 30 deductions including rent relief), CIT (0% for small
  companies, 30% otherwise, plus the 4% development levy), VAT (7.5%, inclusive or exclusive) and
  WHT (5% / 10% by transaction type).
- **Grounded Q&A** — retrieval-augmented answers with numbered citations. The source
  panel shows only the passages the answer actually cites, with their similarity scores.
- **No citation, no claim** — an answer that cites no retrieved passage is withheld and
  reported as ungrounded, whether it is a greeting or a confident claim from the model's
  own knowledge. Below the similarity threshold the LLM is never called at all.
- **Resilient generation** — a transient Gemini 5xx is retried once after a short pause;
  quota errors (429) are not, since retrying them only spends more quota.
- **Clean JSON API** — every endpoint returns either data or `{"error", "message"}`.
  Stack traces never reach the user.
- **Graceful degradation** — missing or invalid credentials produce a readable message
  in the UI, and `/health` stays green so the platform does not kill the container.
- **Reproducible index** — `ingest.py` rebuilds the vector store from source documents.

---

## Architecture

```
                     Browser (templates/index.html)
                 chat box                    calculator tab
                    │                               │
             POST /api/chat                         │
                    │                               │
                    ▼                               │
     chat.py — rule-based router (no AI)            │
     ├─ asks for a figure?   ── read figures ───────┼──► POST /api/calculate/<type>
     │                          missing/unclear?    │              │
     │                          → ask the user      │              ▼
     │                                              └────────► tax.py
     │                                                 fixed rate tables
     │                                                 pure functions, no LLM
     │
     └─ asks for information?  ──────────────────────────► rag.py   (also POST /api/ask)
                                                        1. embed question (Gemini)
                                                        2. search Pinecone namespace
                                                        3. filter by similarity
                                                        4. build cited prompt
                                                        5. generate answer (Gemini)
                                                        6. keep only cited sources
```

`ingest.py` populates the Pinecone index offline and is not part of the request path.

### How the chat router decides

`chat.py` applies these rules in order:

1. **Explicit calculation verbs** — *calculate, compute, work out, estimate* — mean a
   calculation.
2. **Openers that make it a question** — *is, does, who, why, explain, what is the
   difference* — mean information.
3. ***How much*** together with an amount or *pay / owe / due / charged* means a calculation.
4. **A tax followed by *on / for / of* and an amount or *my / this / a*** ("VAT on
   ₦100,000", "tax on my profit") means a calculation.
5. **An amount with none of the above** ("VAT ₦100,000") is unclear, so TaxWiz asks.
6. **Anything else** is information.

For a calculation, the tax is taken from its name (PAYE, CIT, VAT, WHT) or, failing
that, from hints such as *earn* or *salary* (PAYE) and *business* or *turnover* (CIT).
Two taxes at once, a bare "income tax", or no hint at all → TaxWiz asks which one.

Figures are then read from the text: `₦500,000`, `N500,000`, `500k`, `1.5m`,
`2 million naira`. Years ("Act 2025"), counts ("12 months"), percentages and section
numbers are not read as money. Each amount is matched to the nearest keyword
(*pension*, *turnover*, *profit*…). Wherever anything is missing or ambiguous, TaxWiz
asks instead of guessing:

| Tax | Must have | Asked for when missing |
|---|---|---|
| PAYE | gross income **and** whether it is monthly or yearly | yes — a wrong period is a 12× error |
| CIT | annual turnover and assessable profit | yes; monthly figures are also queried |
| VAT | transaction amount | yes; if the message doesn't say, VAT is added on top, and the reply says so |
| WHT | payment amount and payment type | yes |

The only arithmetic in `chat.py` is converting a monthly income to yearly (×12), and the
reply states it. Every figure comes from `tax.py`, which also does all input validation.

An unfinished calculation is returned to the browser as a `pending` object, and the
browser sends it back with the next message. That keeps the server stateless across
gunicorn workers. Because `pending` comes from the client, it is re-validated on every
request.

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
├── chat.py                 # Chat router: calculation vs question, figure extraction
├── tax.py                  # Deterministic tax calculations (no I/O, no LLM)
├── rag.py                  # Embedding → Pinecone retrieval → prompt → LLM
├── ingest.py               # Offline: chunk + embed documents into Pinecone
├── templates/index.html    # Single-page UI (chat + calculator)
├── tests/
│   ├── conftest.py
│   ├── test_tax.py         # Worked examples and input validation
│   ├── test_api.py         # HTTP routing, validation, error contract
│   ├── test_chat.py        # Chat routing, extraction, clarification, no-AI-arithmetic
│   └── test_rag.py         # Pipeline behaviour + live Pinecone checks
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

### The full Act: `nta2025-gemini`

The `nta2025-gemini` namespace holds the full **Nigeria Tax Act, 2025** (Act No. 7,
Official Gazette No. 117, Vol. 112), downloaded from the National Assembly:
<https://nass.gov.ng/documents/download/11249>. The PDF identifies itself as "NIGERIA TAX
ACT 2025 (Edited 1-1-2026)". It has 215 pages, all with extractable text and none of it
reversed. It is not redistributed in this repository. To rebuild the namespace:

```bash
# 1. save the PDF as data/Nigeria-Tax-Act-2025.pdf, then
PINECONE_NAMESPACE=nta2025-gemini python ingest.py --source data
# 2. add the one-definition-per-passage set, copied without re-embedding
PINECONE_NAMESPACE=nta2025-gemini python ingest.py --copy-from wiztax-gemini
```

Step 2 matters. The Act is chunked into 1,200-character windows, so each definition
shares a passage with a dozen others, and definition questions retrieved the wrong term.
With both sets in one namespace, every test question retrieved a relevant passage in the
top five: VAT, VAT liability, deduction at source, PAYE and CIT rates, and three
definitions. The result is 668 passages (441 from the Act plus 227 definitions).
`--copy-from` is only valid between namespaces built with the same embedding model.

### Unreadable passages

Some PDF table layouts come out of text extraction character-reversed — `"snoitpmexE"`
for "Exemptions". Such passages can never answer a question, yet they score well enough
to be retrieved and shown as sources. Every ingestion path skips them, and passages
already indexed can be removed with:

```bash
python ingest.py --prune-garbled --dry-run   # list them
python ingest.py --prune-garbled             # delete them
```

Detection (`looks_reversed` in `ingest.py`) compares how many common English function
words appear forwards ("the", "of") against backwards ("eht", "fo"). On this corpus it
flagged exactly the 33 reversed table dumps out of 260 passages, with no false positives;
those 33 were pruned from `wiztax-gemini`, leaving 227. The originals remain in `wiztax`.

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

### `POST /api/chat`

The conversational endpoint the chat box uses. Send the message, plus the `pending`
object from the previous reply if there was one.

```bash
curl -X POST http://127.0.0.1:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I earn ₦500,000 monthly. Calculate my PAYE."}'
```

Every reply has a `type`, a `reply` text and a `pending` field:

| `type` | Meaning | Extra fields |
|---|---|---|
| `calculation` | Computed by `tax.py` | `tax_type`, `breakdown` (label/value rows), `notice`, `calculation` (raw `tax.py` result) |
| `clarification` | Something is missing or unclear | `pending` — send it back with the next message |
| `answer` | From the Q&A pipeline | `sources`, `grounded` — as `/api/ask` |

```json
{
  "type": "clarification",
  "reply": "Is ₦500,000.00 your monthly or your yearly income?",
  "pending": {"tax_type": "paye", "slots": {"gross_amount": 500000.0}}
}
```

### `POST /api/calculate/<paye|cit|vat|wht>`

```bash
curl -X POST http://127.0.0.1:5000/api/calculate/paye \
  -H "Content-Type: application/json" \
  -d '{"gross_income": 1200000, "pension": 96000, "nhf": 30000}'
```

```json
{
  "tax_type": "paye",
  "gross_income": 1200000.0,
  "pension": 96000.0,
  "nhf": 30000.0,
  "rent_relief": 0.0,
  "eligible_deductions": 126000.0,
  "chargeable_income": 1074000.0,
  "annual_tax": 41100.0,
  "monthly_tax": 3425.0,
  "effective_rate": 3.43,
  "bands": [{ "label": "First ₦800,000", "rate": 0.0, "tax": 0.0, "applied": true },
            { "label": "Next ₦2,200,000", "rate": 15.0, "tax": 41100.0, "applied": true }]
}
```

| Type | Required | Optional |
|---|---|---|
| `paye` | `gross_income` (annual) | `pension`, `nhf`, `nhis`, `mortgage_interest`, `life_assurance`, `annual_rent` |
| `cit` | `turnover`, `assessable_profit`; `fixed_assets` when turnover ≤ ₦100m | `capital_allowances` |
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

`sources` lists only the passages the answer cites. When nothing relevant is retrieved,
or the model's reply cites no passage, `grounded` is `false`, `sources` is empty, and the
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

210 tests covering the tax engine (worked examples verified by hand, band boundaries,
input validation), the HTTP layer (routing, validation, the JSON error contract, no
stack-trace leakage), the RAG pipeline (score filtering, prompt construction, citation
parsing and the no-citation rule, bounded retries, unreadable-passage detection) and the
chat router (`tests/test_chat.py`).

The chat tests run every calculation with Gemini and Pinecone replaced by functions that
**fail the test if called**, and with `tax.calculate` wrapped in a spy. They therefore
prove the figure came from `tax.py` and that no AI was involved, not merely that the
number is right. This was checked by deliberately breaking the code. Routing every message
to Gemini failed 43 tests. Computing VAT inline instead of via `tax.py` failed 10 —
although that copy produced the *correct* VAT, because it skipped `tax.py`'s validation
and calculated tax on ₦0.

`tests/test_rag.py` also contains two live checks, skipped automatically when
`PINECONE_API_KEY` is not set:

- the configured namespace exists, has the right dimension and holds vectors;
- it contains **no unreadable passages**. The first check alone once passed while 12% of
  the corpus was reversed text — counting vectors is not checking content.

What the suite does not establish: that the tax rates match current law, or that real
answers are correct. Generation tests use fake clients by design; answer quality has so
far been checked only by hand on a handful of questions.

---

## Limitations

- **The generation model is pinned one version back.** `gemini-3.6-flash`, not
  `gemini-3.7-flash`. The newer model returns `503 UNAVAILABLE` under free-tier load, and
  `gemini-flash-latest` resolves to it, so both proved unreliable. Switching is a
  one-line `GEMINI_CHAT_MODEL` change — `rag.py` adapts the thinking directive to the
  model generation and retries without it if the model rejects the newer form.

- **The `wiztax-gemini` namespace holds only the Act's definitions.** Its 227 passages
  come from the interpretation section ("Definition of banking: …"), so "What is VAT?"
  has nothing to cite there; TaxWiz says so and offers the calculator instead of answering
  from memory. The `nta2025-gemini` namespace adds the full text of the Act and retrieves
  the right sections for VAT, withholding and rate questions (see "Building the Knowledge
  Base"). Point `PINECONE_NAMESPACE` at it to use it.
- **The chat router understands phrasing it was written for.** It is a set of rules, not
  a language model, so unusual phrasings can be misrouted. The design favours safe
  failures: a calculation request the rules don't recognise goes to the Q&A pipeline,
  which cannot do arithmetic and answers only from cited passages; an unclear amount
  produces a question. It handles one tax per message, monthly or yearly income only
  (not weekly or daily), and English only. It does not read amounts written in words
  ("five hundred thousand").
- **The similarity threshold is not calibrated.** `RAG_MIN_SCORE=0.25` was chosen without
  data. With Gemini embeddings, off-topic questions still score around 0.59, so the
  threshold rarely filters anything; the no-citation rule is doing that work instead.
  Setting it properly needs a labelled set of answerable and unanswerable questions.
- **What the calculator implements from the Nigeria Tax Act 2025, and what it doesn't.**
  The rules were checked against the gazetted text (National Assembly copy, "Edited
  1-1-2026"). Until November 2026 `tax.py` implemented the older, pre-2025 schedule. A
  check against the Act found that PAYE and CIT were wrong, and both were rewritten from
  the Act's text.

  | Rule | Source | Status |
  |---|---|---|
  | PAYE bands: 0% on the first ₦800,000, then 15 / 18 / 21 / 23 / 25% | Fourth Schedule, s. 58 | ✅ implemented |
  | PAYE deductions: pension, NHF, NHIS, mortgage interest, life assurance, rent relief (20% of rent, max ₦500,000) | s. 30 | ✅ implemented |
  | No consolidated relief allowance and no individual minimum tax | absent from the Act | ✅ removed |
  | Exemption for national-minimum-wage earners | s. 58, s. 162(1)(t) | ⚠ not applied — the Act does not state the wage amount; flagged in chat replies |
  | CIT: small company 0%, others 30% | s. 56 | ✅ implemented |
  | Small company: turnover ≤ ₦100m **and** fixed assets ≤ ₦250m | definitions | ✅ implemented |
  | Development levy: 4% of assessable profit, non-small companies | s. 59 | ✅ implemented (resident companies assumed) |
  | 15% minimum effective tax rate (turnover ≥ ₦50bn or MNE groups) | s. 57 | ⚠ flagged, not computed — needs audited accounts |
  | CIT reduced to 25% | s. 56 proviso | ⚠ not applied — only from a date set by presidential order |
  | VAT 7.5% | s. 147 | ✅ matches |
  | WHT rates | not in this Act ("prescribed by Nigeria Tax Administration Act") | ⚠ unverified |

  This is still an estimate. The Act is long, and other provisions — exemptions, special
  regimes, sector rules — are not modelled.
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
