# TaxWiz 🇳🇬⚖️
> An intelligent Nigerian Tax Law assistant powered by RAG (Retrieval-Augmented Generation)

TaxWiz is a conversational AI chatbot that answers questions about Nigerian tax law, grounded exclusively in the **Nigerian Tax Act 2025**. It combines vector search (Pinecone) with a local language model (Qwen) to deliver accurate, context-aware responses — plus a built-in tax calculator for PAYE, CIT, VAT, and WHT computations.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-2.2-lightgrey)
![Pinecone](https://img.shields.io/badge/Vector_DB-Pinecone-purple)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Features

- 💬 **AI Tax Assistant** — Ask questions about Nigerian tax law in plain English
- 🧮 **Tax Calculator** — Compute PAYE, Company Income Tax, VAT, and Withholding Tax
- 📚 **RAG Pipeline** — Answers grounded in the Nigerian Tax Act 2025 via Pinecone vector search
- 🤖 **Local LLM** — Runs on Qwen2.5 (HuggingFace) with no cloud dependency for generation
- 🎨 **Modern UI** — Clean dark-themed chat interface with typing indicators and timestamps

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Vector Database | Pinecone |
| Embeddings | Qwen3-Embedding-0.6B (HuggingFace) |
| Language Model | Qwen2.5-0.5B-Instruct (HuggingFace, local) |
| Document Parsing | Azure Form Recognizer |
| Frontend | HTML, CSS, JavaScript (jQuery) |

---

## Project Structure

```
ChatOps/
├── app/
│   ├── backend/
│   │   ├── app.py                  # Main Flask application
│   │   ├── approaches/             # RAG approach implementations
│   │   │   ├── approach.py
│   │   │   ├── chatreadretrieveread.py
│   │   │   ├── retrievethenread.py
│   │   │   ├── readretrieveread.py
│   │   │   └── readdecomposeask.py
│   │   ├── langchainadapters.py    # LangChain callback handler
│   │   ├── lookuptool.py           # CSV lookup tool
│   │   └── text.py                 # Text utilities
│   └── frontend/
│       ├── static/                 # CSS and images
│       └── templates/
│           └── index.html          # Main UI (chat + calculator)
├── prepdocs.py                     # Azure Search data ingestion pipeline
├── pineconemethod.py               # Pinecone data ingestion pipeline
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- A [Pinecone](https://www.pinecone.io/) account and API key
- An [OpenAI](https://platform.openai.com/) API key (optional, for embeddings fallback)
- ~3GB free disk space (for local Qwen models)

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/sadiqmuhd/Chatops.git
cd Chatops
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install flask python-dotenv openai pinecone-client sentence-transformers transformers torch accelerate
```

**4. Set up environment variables**

Create a `.env` file in `app/backend/`:
```env
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
```

**5. Run the app**
```bash
cd app/backend
flask --app app.py run
```

Open your browser at `http://127.0.0.1:5000`

---

## Data Ingestion

To populate Pinecone with your own documents, place PDF files in the `data/` folder and run:

```bash
python pineconemethod.py
```

This will:
1. Extract text from PDFs using Azure Form Recognizer
2. Generate embeddings using the configured embedding model
3. Upsert vectors into your Pinecone index

---

## Tax Calculator

The built-in calculator (accessible from the sidebar) supports:

| Tax Type | Details |
|---|---|
| **PAYE** | Graduated rates 7%–24%, with CRA, pension & NHF deductions |
| **Company Income Tax** | 0% (small), 20% (medium), 30% (large) based on turnover |
| **VAT** | 7.5% — supports both inclusive and exclusive calculations |
| **Withholding Tax** | 5%–10% depending on transaction type |

All calculations run entirely in the browser — no backend call required.

---

## Architecture

```
User Query
    │
    ▼
Embed query (Qwen3-Embedding-0.6B)
    │
    ▼
Vector search → Pinecone (taxwiz3 index)
    │
    ▼
Retrieve top-k relevant chunks from Nigerian Tax Act 2025
    │
    ▼
Build prompt with context + system instructions
    │
    ▼
Generate response (Qwen2.5-0.5B-Instruct, local)
    │
    ▼
Return answer to user
```

---

## Notes

- The first query after startup will be slow (~30–60s) as models are downloaded and cached locally
- Subsequent queries are faster (~10–20s on CPU)
- For best performance, a machine with 16GB+ RAM and a GPU is recommended
- This project is for educational/demonstration purposes. Always verify tax calculations with a certified tax professional.

---

## License

MIT License — feel free to use and modify for your own projects.

---

*Built with ❤️ — Nigerian Tax Act 2025*
