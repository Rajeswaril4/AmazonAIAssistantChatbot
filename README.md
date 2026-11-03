# 🛒 Amazon AI Assistant

A shopping and customer-service chatbot built with **RAG + a tool-calling LLM agent**. It recommends products from real Amazon data, answers policy questions from Amazon help pages with sources, and handles demo orders (track, return, cancel) behind a login.

> Demo project. Orders are sample data, because Amazon has no public API for real customer orders.

## Features

| Feature | How it works |
|---|---|
| Product search | Semantic search over products and reviews, with price and rating filters (`under 50`, `between 20 and 40`, `₹2000`) |
| Product cards | Photo box (same size for every card), rating, title, price, link to Amazon |
| Review summaries | Pros and cons come only from retrieved review snippets |
| Policy answers | Returns, refunds, A-to-z Guarantee, privacy, conditions of use, with cited source passages |
| Follow-up memory | "cheaper", "better rated", "compare the first two" reuse the previous search |
| Compare | Side-by-side table of 2-4 products |
| Login | Sign up / log in with salted PBKDF2 password hashes, or continue as a guest |
| Chat history | Saved per user and listed in the sidebar: reopen, delete, or start a new chat |
| Orders | Show orders, track, return (30-day window) and cancel (before shipping), each with a Confirm button |
| Human hand-off | Creates a support ticket when you ask for a person |
| "I don't know" | A reranker score cutoff stops irrelevant matches ("best laptop") from being recommended |

## Architecture

```
Streamlit chat (login, history, product cards)
        │
        ▼
  LLM with tools  ← OpenRouter API (OpenAI-compatible)
   ├─ search_products   → ChromaDB: products + reviews  (bge embeddings → cross-encoder rerank)
   ├─ compare_products  → ChromaDB
   ├─ search_policy     → ChromaDB: policy chunks
   ├─ get_orders / track_order
   ├─ request_return / request_cancel   → only PREPARES the action; the user presses Confirm
   └─ escalate_to_human
        │
        ▼
 SQLite (orders.db): users, orders, tickets, chat messages
```

Design rules:
- Only the Confirm button changes order data, so the model cannot claim an action happened when it did not.
- The model answers from tool results, never from its own knowledge.
- Retrieval: fetch 15-20 → drop weak vector matches → rerank → keep the best → cutoff (`MIN_RERANK`).

## Project structure

```
app.py             Streamlit UI: login, sidebar, chat, product cards
agent.py           Tools, system prompt and agent loop (no Streamlit)
auth.py            Accounts and saved chat history
orders.py          Demo order system (SQLite)
config.py          Categories, models, thresholds, LLM settings
prepare_data.py    Step 1: shrink raw Amazon files to the top products and reviews
build_index.py     Step 2 (Colab GPU): embed everything into ChromaDB
extract_images.py  Step 3: collect product image URLs into images.json
fetch_policies.py  Download the help pages listed in policy_urls.txt
policy_urls.txt    Policy page list (name | url)
requirements.txt
```

## Setup

### 1. Get the data
- Download the product metadata (`meta_<Category>.jsonl.gz`) and, optionally, reviews (`<Category>.jsonl.gz`) from the [Amazon Reviews 2023 dataset](https://amazon-reviews-2023.github.io/) into `data/`. Categories are listed in `config.py`.
- Download the policy pages:
  ```bash
  pip install trafilatura requests
  python fetch_policies.py        # writes data/amazon/*.md; copy blocked pages by hand
  ```

### 2. Build the index (Google Colab, T4 GPU recommended)
```bash
pip install -q chromadb sentence-transformers pandas pyarrow
python prepare_data.py            # → data/processed/*.parquet
python build_index.py             # → db/   (delete the old db/ first if you change EMBED_MODEL)
python extract_images.py          # → images.json (optional, for product photos)
```
Download `db/` and `images.json` and put them next to `app.py`.

### 3. Run the app
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
echo 'OPENROUTER_API_KEY=your-key-here' > .env
streamlit run app.py
```
Open http://localhost:8501. The first start downloads the embedding and reranker models.

## Configuration (`config.py`)

| Setting | Meaning |
|---|---|
| `CATS` | Product categories to index |
| `MAX_PRODUCTS`, `MAX_REVIEWS` | Products per category, reviews per product |
| `EMBED_MODEL`, `RERANK_MODEL` | Embedding and reranking models (rebuild the index after changing the embedder) |
| `MIN_RERANK` | Reranker cutoff. Raise it if irrelevant products appear, lower it if good questions get "no match" |
| `LLM_BASE_URL`, `LLM_MODEL`, `LLM_FALLBACK` | Any OpenAI-compatible API with tool calling (OpenRouter by default; a local Ollama URL also works) |

## Example questions

- Products: `ukulele under 50` → `what about a cheaper one?` → `compare the first two`
- Reviews: `any complaints about baby monitors?`
- Policies: `how long do refunds take?`, `what is the A-to-z Guarantee?`
- Orders (log in first): `show my orders`, `track ORD1002`, `cancel ORD1001`, `return ORD1004`
- Other: `I want to talk to a person`, `best laptop` (should say no match)

## Notes and limitations

- Each new account gets 5 sample orders in different states.
- Prices come from the dataset (USD) and may be out of date. There is no live stock or tracking.
- Only products in the configured categories are known.
- `orders.db` holds accounts, orders and chat history; it is not committed to Git. The login is for demo use: a real deployment needs password reset, rate limiting and HTTPS.
- Answer quality depends on the LLM's tool-calling ability.

## Tech stack

Python · Streamlit · ChromaDB · sentence-transformers (BGE embeddings, MS MARCO cross-encoder) · OpenRouter / OpenAI SDK · SQLite · pandas
