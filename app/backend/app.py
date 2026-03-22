import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import time
import torch
import logging
import sys
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
from flask import Flask, render_template, request
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ── Model cache globals ────────────────────────────────────────────────────────
_embedding_model = None
_qwen_model = None
_qwen_tokenizer = None
_qwen_model_name_loaded = None

def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model for the first time...")
        _embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        logger.info("Embedding model loaded and cached.")
    return _embedding_model

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="../frontend/templates", static_folder="../frontend/static")

system_prompt = """
You are a Nigerian Tax Law assistant.

You MUST answer ONLY using the provided context.
If the answer is not explicitly contained in the context, respond with:
"This information is not contained in the Nigerian Tax Act 2025."

Do NOT use prior knowledge.
Do NOT guess.
Do NOT answer questions outside Nigerian tax law.

Context:
{context}

Question: {question}
"""

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    logger.info("Received request for bot response")
    userText = request.args.get('msg')
    logger.info(f"Received user query: {userText}")

    # Step 1: Embed the query using cached model
    emb = generate_embeddings([userText], hf_model=get_embedding_model())

    # Step 2: Retrieve relevant chunks from Pinecone
    context = retrieve_relevant_chunks(emb[0], 'taxwiz3', top_k=2)
    logger.info(f"Retrieved context: {context}")

    # Step 3: Build messages
    messages = [
        {"role": "system", "content": system_prompt.format(context=context, question=userText)},
        {"role": "user", "content": userText}
    ]

    # Step 4: Generate response
    resp = generate_huggingface_response(messages)
    logger.info(f"Generated response: {resp}")
    return resp


# ── Client helpers ─────────────────────────────────────────────────────────────
def get_openai_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)

def get_pinecone_client() -> Pinecone:
    return Pinecone(api_key=PINECONE_API_KEY)

def get_pinecone_index(index_name: str):
    pc = get_pinecone_client()
    return pc.Index(index_name)


# ── Retrieval ──────────────────────────────────────────────────────────────────
def retrieve_relevant_chunks(vec, index_name: str, top_k: int = 2) -> str:
    index = get_pinecone_index(index_name)
    results = index.query(vector=vec, top_k=top_k, include_metadata=True)
    match_val = [match['metadata']['text'] for match in results['matches']]
    return "\n".join(match_val)


# ── Embeddings ─────────────────────────────────────────────────────────────────
def generate_embeddings(
    text_chunks: List[str],
    provider: str = "huggingface",
    hf_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    openai_model_name: str = "text-embedding-3-small",
    hf_model: Optional[SentenceTransformer] = None,
) -> List[List[float]]:
    if not text_chunks:
        return []

    if provider.lower() == "huggingface":
        model = hf_model or SentenceTransformer(hf_model_name)
        embeddings = model.encode(text_chunks, convert_to_numpy=True)
        return embeddings.tolist()

    elif provider.lower() == "openai":
        client = get_openai_client()
        embeddings = []
        for chunk in text_chunks:
            response = client.embeddings.create(model=openai_model_name, input=chunk)
            embeddings.append(response.data[0].embedding)
        return embeddings

    else:
        raise ValueError("provider must be either 'huggingface' or 'openai'")


# ── Generation ─────────────────────────────────────────────────────────────────
def generate_huggingface_response(
    messages: List[dict],
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens: int = 200,
    do_sample: bool = False,
) -> str:
    global _qwen_model, _qwen_tokenizer, _qwen_model_name_loaded

    if _qwen_model is None or _qwen_tokenizer is None or _qwen_model_name_loaded != model_name:
        logger.info(f"Loading generation model: {model_name}")
        _qwen_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _qwen_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None,
        )
        _qwen_model.eval()
        _qwen_model_name_loaded = model_name
        logger.info("Generation model loaded and cached.")

    text = _qwen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = _qwen_tokenizer([text], return_tensors="pt")

    with torch.no_grad():
        outputs = _qwen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = _qwen_tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False)
