CATS = [
    "Handmade_Products", "Appliances", "Baby_Products", "Cell_Phones_and_Accessories",
    "Pet_Supplies", "Musical_Instruments", "Industrial_and_Scientific", "Software",
]                                       # each needs data/meta_<name>.jsonl.gz
RAW, OUT, POLICY, DB = "data", "data/processed", "data/amazon", "db"
MAX_PRODUCTS, MAX_REVIEWS = 3000, 15    # per category / per product

# --- retrieval ---
EMBED_MODEL = "BAAI/bge-small-en-v1.5"                    # REBUILD the index after changing
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RETRIEVE_N, KEEP_N = 15, 4                                # fetch 15, rerank, keep best 4
MAX_DIST = 0.8                                            # loose vector pre-filter (bge distances are compressed)
MIN_RERANK = 0.0                                          # reranker score cutoff: raise = stricter, lower = more answers
INR_PER_USD = 85                                          # used when the user writes a budget in rupees

# --- LLM via API (OpenRouter) ---
LLM_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "deepseek/deepseek-chat"
LLM_FALLBACK = "meta-llama/llama-3.1-70b-instruct"        # used if the main model fails