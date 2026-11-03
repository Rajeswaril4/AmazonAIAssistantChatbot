"""Tool-calling agent: tools, system prompt and agent loop. No Streamlit here, so it can be tested on its own."""
import json, re, datetime as dt
import orders
from config import RETRIEVE_N, KEEP_N, MAX_DIST, MIN_RERANK, INR_PER_USD

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
    def retrieve(self, col, query, n=RETRIEVE_N, keep=KEEP_N, **kw):
        qe = self.embed(query)
        r = col.query(query_embeddings=qe, n_results=n, include=["documents", "metadatas", "distances"], **kw)
        hits = [(i, d, m) for i, d, m, s in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0]) if s <= MAX_DIST]
        if not hits: return [], qe
        scores = self.rerank([(query, d) for _, d, _ in hits])
        ranked = sorted(zip(scores, hits), key=lambda x: -x[0])
        return [h for s, h in ranked if s >= MIN_RERANK][:keep], qe

    # ---- tools ----
    def search_products(self, query, min_price=None, max_price=None, min_rating=None):
        if min_price is None and max_price is None: min_price, max_price = parse_budget(self.question)
        conds = ([{"price": {"$gte": float(min_price)}}] if min_price is not None else []) + \
                ([{"price": {"$lte": float(max_price)}}] if max_price is not None else []) + \
                ([{"rating": {"$gte": float(min_rating)}}] if min_rating is not None else [])
        where = None if not conds else conds[0] if len(conds) == 1 else {"$and": conds}
        hits, qe = self.retrieve(self.cols["products"], query, n=20, keep=5, **({"where": where} if where else {}))
        self.ctx["search"] = dict(query=query, min_price=min_price, max_price=max_price, min_rating=min_rating,
                                  shown=[f"{i}: {m['title'][:40]} ${m['price']} {m['rating']}★" for i, _, m in hits])
        if not hits: return {"products": [], "note": "No matching products found" + (" with those filters." if where else ".")}
        ids = [i for i, _, _ in hits]
        revs = {}
        if self.cols.get("reviews") is not None:
            r = self.cols["reviews"].query(query_embeddings=qe, n_results=20, where={"asin": {"$in": ids}}, include=["documents", "metadatas"])
            for d, m in zip(r["documents"][0], r["metadatas"][0]):
                if len(revs.setdefault(m["asin"], [])) < 2: revs[m["asin"]].append(f"({m['rating']:.0f}★) {d[:200]}")
        items = [dict(id=i, title=m["title"], price=m["price"], rating=m["rating"], ratings=m["n"], category=m.get("category", "")) for i, _, m in hits]
        self.ctx["cards"] += items
        return {"products": [dict(it, reviews=revs.get(it["id"], [])) for it in items]}

    def compare_products(self, ids):
        got = self.cols["products"].get(ids=list(ids)[:4], include=["metadatas"])
        rows = [{"ID": i, "Title": m["title"][:70], "Price ($)": m["price"], "Rating": m["rating"], "Ratings": m["n"], "Category": m.get("category", "")}
                for i, m in zip(got["ids"], got["metadatas"])]
        if not rows: return {"error": "None of those ids exist. Search for the products first."}
        self.ctx["compare"] = rows
        revs = []
        if self.cols.get("reviews") is not None:
            r = self.cols["reviews"].get(where={"asin": {"$in": [x["ID"] for x in rows]}}, limit=12, include=["documents", "metadatas"])
            revs = [f"[{m['asin']}] ({m['rating']:.0f}★) {d[:200]}" for d, m in zip(r["documents"], r["metadatas"])]
        return {"products": rows, "review_snippets": revs}

    def search_policy(self, question):
        if self.cols.get("policy") is None: return {"passages": [], "note": "No policy documents indexed."}
        hits, _ = self.retrieve(self.cols["policy"], question)
        if not hits: return {"passages": [], "note": "Nothing relevant found in the policy documents."}
        start, out = len(self.ctx["sources"]) + 1, []
        for n, (_, d, m) in enumerate(hits, start):
            out.append(dict(n=n, source=m["source"], section=m.get("section", ""), text=d))
            self.ctx["sources"].append(f"[{n}] {m['source']}  ·  {m.get('section', '')}\n{d}")
        return {"passages": out}

    def _need_user(self):
        return None if self.user else {"error": "NOT_SIGNED_IN: ask the user to log in (Log out, then Log in / Sign up in the sidebar)."}

    def get_orders(self):
        if e := self._need_user(): return e
        rows = orders.list_orders(self.user)
        self.ctx["orders"] = [{"Order": o["order_id"], "Product": o["title"][:60], "Price ($)": o["price"], "Status": o["status"],
                               "Ordered": o["order_date"], "Delivered / ETA": o["delivered_date"] or o["eta"]} for o in rows]
        return {"orders": self.ctx["orders"]}

    def track_order(self, order_id):
        if e := self._need_user(): return e
        o = orders.get_order(self.user, order_id)
        return orders.track(o) if o else {"error": f"ORDER_NOT_FOUND: no order {order_id} on this account."}

    def request_return(self, order_id, reason="not specified"):
        if e := self._need_user(): return e
        o = orders.get_order(self.user, order_id)
        if not o: return {"error": f"ORDER_NOT_FOUND: no order {order_id} on this account."}
        ok, why = orders.return_check(o)
        if not ok: return {"eligible": False, "reason": why}
        self.ctx["pending"] = dict(type="return", order_id=o["order_id"], reason=reason, user=self.user,
                                   summary=f"Return “{o['title'][:60]}” (order {o['order_id']}). Reason: {reason}")
        return {"eligible": True, "detail": why, "status": "AWAITING_USER_CONFIRMATION",
                "note": "Tell the user to press the Confirm button below the chat. Nothing has been returned yet."}

    def request_cancel(self, order_id):
        if e := self._need_user(): return e
        o = orders.get_order(self.user, order_id)
        if not o: return {"error": f"ORDER_NOT_FOUND: no order {order_id} on this account."}
        ok, why = orders.cancel_check(o)
        if not ok: return {"cancellable": False, "reason": why}
        self.ctx["pending"] = dict(type="cancel", order_id=o["order_id"], reason="", user=self.user,
                                   summary=f"Cancel “{o['title'][:60]}” (order {o['order_id']})")
        return {"cancellable": True, "status": "AWAITING_USER_CONFIRMATION",
                "note": "Tell the user to press the Confirm button below the chat. Nothing has been cancelled yet."}

    def escalate_to_human(self, reason):
        self.ctx["ticket"] = orders.create_ticket(self.user or "guest", reason)
        return {"ticket_id": self.ctx["ticket"], "note": "A human agent will follow up (demo ticket)."}

    def call(self, name, args):
        self.ctx["tools"].append(name)
        try:
            if name not in {t["function"]["name"] for t in TOOLS}: return {"error": f"unknown tool {name}"}
            return getattr(self, name)(**(args or {}))
        except Exception as e:
            return {"error": f"{name} failed: {e!r}"}

def run_agent(complete, tb, system, history, question, max_steps=6):
    """complete(messages, tools) -> assistant message object with .content and .tool_calls."""
    msgs = [{"role": "system", "content": system}] + history + [{"role": "user", "content": question}]
    for _ in range(max_steps):
        m = complete(msgs, TOOLS)
        if not m.tool_calls: return m.content or ""
        msgs.append({"role": "assistant", "content": m.content or "", "tool_calls": [
            {"id": t.id, "type": "function", "function": {"name": t.function.name, "arguments": t.function.arguments or "{}"}} for t in m.tool_calls]})
        for t in m.tool_calls:
            try: args = json.loads(t.function.arguments or "{}")
            except Exception: args = {}
            msgs.append({"role": "tool", "tool_call_id": t.id, "content": json.dumps(tb.call(t.function.name, args), default=str)[:5000]})
    return "Sorry, I couldn't finish that request. Please try rephrasing it."