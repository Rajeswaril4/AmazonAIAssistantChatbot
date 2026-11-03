"""Tool-calling agent: tools, system prompt and agent loop. No Streamlit here, so it can be tested on its own."""
import json, re, datetime as dt
import orders
from config import RETRIEVE_N, KEEP_N, MAX_DIST, MIN_RERANK, INR_PER_USD, CATS

UNDER_RE = re.compile(r"(?:under|below|less than|within|up to|upto|max(?:imum)?|<=?)\s*([$₹]|rs\.?|inr)?\s*(\d[\d,]*\.?\d*)", re.I)
OVER_RE = re.compile(r"(?:over|above|more than|at least|min(?:imum)?|>=?)\s*([$₹]|rs\.?|inr)?\s*(\d[\d,]*\.?\d*)", re.I)
BETWEEN_RE = re.compile(r"between\s*([$₹]|rs\.?|inr)?\s*(\d[\d,]*\.?\d*)\s*(?:and|to|-)\s*([$₹]|rs\.?|inr)?\s*(\d[\d,]*\.?\d*)", re.I)

def to_usd(sym, num):
    v = float(num.replace(",", ""))
    return v / INR_PER_USD if sym and sym.lower() != "$" else v

def parse_budget(text):
    """Backup budget parser -> (min_usd, max_usd)."""
    if m := BETWEEN_RE.search(text): return to_usd(m[1] or m[3], m[2]), to_usd(m[3] or m[1], m[4])
    lo = hi = None
    if m := UNDER_RE.search(text): hi = to_usd(m[1], m[2])
    if m := OVER_RE.search(text): lo = to_usd(m[1], m[2])
    return lo, hi

def _fn(name, desc, props, required=()):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(required)}}}

TOOLS = [
    _fn("search_products", "Search the product catalogue (with review snippets). Use for ANY request to find, recommend or ask about products or what reviewers say.",
        {"query": {"type": "string", "description": "standalone product search query, no price words; for follow-ups reuse the previous search's query"},
         "min_price": {"type": "number", "description": "USD, optional"}, "max_price": {"type": "number", "description": "USD, optional"},
         "min_rating": {"type": "number", "description": "minimum star rating, optional (e.g. 4.5 for 'better rated')"}}, ["query"]),
    _fn("compare_products", "Compare 2-4 products by their ids (from earlier search results).",
        {"ids": {"type": "array", "items": {"type": "string"}}}, ["ids"]),
    _fn("search_policy", "Search Amazon help pages: returns, refunds, A-to-z Guarantee, privacy, conditions of use, missing packages, replacements.",
        {"question": {"type": "string"}}, ["question"]),
    _fn("get_orders", "List the signed-in user's orders.", {}),
    _fn("track_order", "Tracking and status of one of the user's orders.", {"order_id": {"type": "string"}}, ["order_id"]),
    _fn("request_return", "Check return eligibility and PREPARE a return. The user must press a Confirm button; nothing is returned by this call.",
        {"order_id": {"type": "string"}, "reason": {"type": "string"}}, ["order_id"]),
    _fn("request_cancel", "Check cancel eligibility and PREPARE a cancellation. The user must press a Confirm button; nothing is cancelled by this call.",
        {"order_id": {"type": "string"}}, ["order_id"]),
    _fn("escalate_to_human", "Create a support ticket for a human agent when the user asks for one or the problem cannot be solved with the tools.",
        {"reason": {"type": "string"}}, ["reason"]),
]

def build_system(user_label, categories, last_search=None):
    prev = ("Previous product search (for follow-ups): " + json.dumps(last_search, default=str) + ". ") if last_search else "No product search yet in this chat. "
    return (f"You are the Amazon AI Assistant (demo). Today is {dt.date.today().isoformat()}. Signed-in user: {user_label or 'nobody (guest)'}. "
            f"Product categories you know: {categories}. Always use tools; never answer from your own knowledge.\n"
            f"- {prev}For follow-ups such as 'cheaper', 'better rated', 'more options' or 'show others', call search_products again with the SAME query as the previous search "
            "and adjust the filters (cheaper: max_price below the lowest price shown; better rated: min_rating above the highest rating shown, or 4.5; "
            "more options: same query). Never ask the user to repeat the product when a previous search exists.\n"
            "- Product requests: call search_products. Budgets: pass min_price/max_price in USD (convert rupees at 85 per USD). "
            "Recommend the best 3 that really match; if none match, say so in one sentence. Name each by a short title plus [id]. Give price, rating, and pros/cons ONLY from the "
            "review snippets, paraphrased in under 15 words; if asked about complaints, focus on cons. Never add general knowledge about a product type. Prices are USD.\n"
            "- Comparisons: call compare_products with ids from earlier results.\n"
            "- Policy questions: call search_policy, answer only from the passages, cite [1], [2]; if not covered say you don't know.\n"
            "- Orders: need a signed-in user (otherwise ask them to log in). Use get_orders / track_order. "
            "For returns or cancellations call request_return / request_cancel; these only PREPARE the action and the user confirms with a button. "
            "Never say something was returned or cancelled. If not eligible, explain why using the tool's reason.\n"
            "- If the user asks for a human or you cannot help with the tools, call escalate_to_human.\n"
            "- Greetings and questions about what you can do: answer briefly without tools. Unrelated general-knowledge questions: politely decline. Be concise.")

class Toolbox:
    def __init__(self, cols, embed, rerank):
        self.cols, self.embed, self.rerank = cols, embed, rerank

    def start(self, user, question):
        self.user, self.question = user, question
        self.ctx = dict(cards=[], sources=[], compare=None, orders=None, pending=None, ticket=None, tools=[], search=None)

    # ---- retrieval ----
    def retrieve(self, col, query, n=RETRIEVE_N, keep=KEEP_N, strict=True, **kw):
        qe = self.embed(query)
        r = col.query(query_embeddings=qe, n_results=n, include=["documents", "metadatas", "distances"], **kw)
        # Strict retrieval applies similarity thresholds. Relaxed retrieval is only a fallback.
        hits = [
            (i, d, m) for i, d, m, distance
            in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])
            if (distance <= MAX_DIST or not strict)
        ]
        if not hits:
            return [], qe
        scores = self.rerank([(query, d) for _, d, _ in hits])
        ranked = sorted(zip(scores, hits), key=lambda x: -x[0])
        if strict:
            return [h for score, h in ranked if score >= MIN_RERANK][:keep], qe
        return [h for _, h in ranked][:keep], qe

    # ---- tools ----
    def search_products(self, query, min_price=None, max_price=None, min_rating=None):
        if min_price is None and max_price is None:
            min_price, max_price = parse_budget(self.question)

        def norm(value):
            return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
