"""Amazon AI Assistant: product search, reviews, policies, demo orders, and chat."""

# Import Streamlit first so the page can show diagnostics before loading heavy AI libraries.
import os
import re
import time
import json
import html
import uuid
import traceback
import streamlit as st

st.set_page_config(page_title="Amazon AI Assistant", page_icon="🛒", layout="wide")
st.title("🛒 Amazon AI Shopping Intelligence Assistant")
st.caption("Starting the application. The first startup can take several minutes while AI models download.")

# Load dependencies after showing the page, so import errors are visible in the app.
try:
    import chromadb
    import pandas as pd
    from dotenv import load_dotenv
    from openai import OpenAI
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import config, orders, auth
    from config import DB, EMBED_MODEL, RERANK_MODEL, LLM_BASE_URL, LLM_MODEL
    from agent import Toolbox, build_system, run_agent
    LLM_FALLBACK = getattr(config, "LLM_FALLBACK", None)
except Exception as e:
    st.error("The application could not import one or more dependencies.")
    st.exception(e)
    st.code(traceback.format_exc())
    st.stop()

load_dotenv()
try:
    API_KEY = (
        os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPENROUTER_KEY")
        or st.secrets.get("OPENROUTER_API_KEY", "")
    ).strip().strip("\"'“”‘’")
except Exception:
    API_KEY = (os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_KEY") or "").strip().strip("\"'“”‘’")

if not API_KEY:
    st.error("OPENROUTER_API_KEY is missing. Add it in Streamlit Cloud → App settings → Secrets.")
    st.info('Add this entry to Secrets: OPENROUTER_API_KEY = "your_actual_key"')
    st.stop()

llm = OpenAI(base_url=LLM_BASE_URL, api_key=API_KEY)

DB_PATH = os.path.abspath("db_deploy")

@st.cache_resource(show_spinner=False)
def load_database(db_path):
    """Open the local ChromaDB directory and return the collections that exist."""
    db = chromadb.PersistentClient(path=db_path)
    cols, errors = {}, {}
    for name in ("policy", "products", "reviews"):
        try:
            cols[name] = db.get_collection(name)
        except Exception as e:
            cols[name] = None
            errors[name] = repr(e)
    return cols, errors

@st.cache_resource(show_spinner=False)
def load_models(embed_name, rerank_name):
    """Load cached embedding and reranking models."""
    embedding_model = SentenceTransformer(embed_name)
    reranking_model = CrossEncoder(rerank_name)
    return embedding_model, reranking_model

@st.cache_data(show_spinner=False)
def load_images():
    """images.json is optional; it maps product IDs to image URLs."""
    try:
        with open("images.json", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

# Initialize the database first. Do not spend time downloading AI models if the index is missing.
try:
    with st.status("Checking the product database...", expanded=True) as startup_status:
        st.write(f"Database path: `{DB_PATH}`")
        cols, errors = load_database(DB_PATH)
        if cols.get("products") is None:
            st.error(f"The required ChromaDB 'products' collection was not found. Details: {errors.get('products', 'No collection available')}")
            st.warning("The `db/` directory must be available beside app.py in the deployed app. Your .gitignore excludes db/, so it will not be pushed by Git unless you deliberately change that setup.")
            st.stop()

        st.write("Product collection found. Loading embedding and reranker models; first startup may take several minutes...")
        model, reranker = load_models(EMBED_MODEL, RERANK_MODEL)
        startup_status.update(label="Database and AI models loaded", state="complete", expanded=False)
except Exception as e:
    st.error("Application startup failed while opening the database or loading AI models.")
    st.exception(e)
    st.code(traceback.format_exc())
    st.stop()

IMAGES = load_images()

def complete(msgs, tools=None, max_tokens=800):
    """One LLM call: retry on rate limit, shrink max_tokens when credit is low (402), fall back to a second model."""
    err = None
    for name in [LLM_MODEL] + ([LLM_FALLBACK] if LLM_FALLBACK else []):
        for attempt in range(3):
            try:
                kw = {"tools": tools} if tools else {}
                return llm.chat.completions.create(model=name, max_tokens=max_tokens, temperature=0.2, messages=msgs, **kw).choices[0].message
            except Exception as e:
                err, s = e, str(e)
                if "402" in s and (m := re.search(r"can only afford (\d+)", s)) and int(m[1]) > 250:
                    max_tokens = int(m[1]) - 20; continue                  # low credit: ask for fewer tokens and retry
                if "429" in s: time.sleep(2 ** (attempt + 1)); continue
                break
    hint = " Your OpenRouter credit is too low: add credit at https://openrouter.ai/settings/credits." if "402" in str(err) else ""
    raise RuntimeError(f"LLM request failed: {err}{hint}")

def sample_products():
    g = cols["products"].get(limit=400, include=["metadatas"])
    return [dict(asin=i, title=m["title"], price=m["price"]) for i, m in zip(g["ids"], g["metadatas"])]

orders.init_db(); auth.init()
tb = Toolbox(cols, lambda t: model.encode([t], normalize_embeddings=True).tolist(),
             lambda pairs: reranker.predict(pairs))

S = st.session_state
for k, v in dict(auth=False, user=None, history=[], pending=None, last_search=None, chat_id=uuid.uuid4().hex).items(): S.setdefault(k, v)

def fresh(**kw):                       # switch to another (or new) chat and redraw
    S.update(dict(history=[], pending=None, last_search=None, chat_id=uuid.uuid4().hex) | kw); st.rerun()

def add(**m):                          # show a message and save it to the user's history
    S.history.append(m)
    if S.user: auth.save(S.user, S.chat_id, m)

# ---------- login page ----------
if not S.auth:
    st.title("🛒 Amazon AI Assistant")
    st.caption("Log in to keep your orders and chat history, or continue as a guest.")
    for tab, label, fn in zip(st.tabs(["Log in", "Sign up"]), ["Log in", "Sign up"], [auth.login, auth.register]):
        with tab, st.form(label):
            u, p = st.text_input("Username"), st.text_input("Password", type="password")
            if st.form_submit_button(label):
                user, err = fn(u, p)
                if err: st.error(err)
                else:
                    orders.seed_user(user, sample_products)
                    S.auth, S.user = True, user; fresh()
    if st.button("Continue as guest"): S.auth, S.user = True, None; fresh()
    st.stop()
user = S.user

# ---------- sidebar ----------
with st.sidebar:
    st.subheader("Account")
    st.write(f"👤 **{user}**" if user else "👤 Guest (orders and history are not saved)")
    if st.button("Log out" if user else "Log in / Sign up"): S.auth, S.user = False, None; fresh()
    if st.button("➕ New chat"): fresh()
    if user:
        st.subheader("Chat history")
        for cid, title in auth.chats(user):
            a, b = st.columns([5, 1])
            if a.button(("▶ " if cid == S.chat_id else "") + title, key=f"o{cid}", use_container_width=True): fresh(history=auth.load(user, cid), chat_id=cid)
            if b.button("🗑", key=f"d{cid}"): auth.delete_chat(user, cid); fresh() if cid == S.chat_id else st.rerun()
        if st.button("Reset my demo orders"): orders.reset(user, sample_products); S.pending = None; st.rerun()
    st.subheader("What I know")
    st.write("Products: " + ", ".join(c.replace("_", " ") for c in config.CATS))
    st.write("Policies: returns, refunds, A-to-z Guarantee, privacy, conditions of use")
    st.caption("Prices in USD (₹ budgets are converted). Try: 'ukulele under 50', then 'what about a cheaper one?', 'compare the first two', 'show my orders'.")
    if not IMAGES: st.caption("No images.json found: product cards show without photos.")
if S.pending and S.pending["user"] != user: S.pending = None

# ---------- rendering ----------
st.markdown("""<style>
.pimg{height:170px;display:flex;align-items:center;justify-content:center;background:#fff;border-radius:8px;margin-bottom:8px;font-size:48px}
.pimg img{max-width:100%;max-height:100%;object-fit:contain}
.ptitle{font-weight:600;line-height:1.45em;height:2.9em;overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical}
</style>""", unsafe_allow_html=True)

def show_cards(cards):
    for row in range(0, len(cards), 3):
        for col, c in zip(st.columns(3), cards[row:row + 3]):       # always 3 equal columns, so every card has the same width
            with col.container(border=True):
                img = IMAGES.get(c["id"])
                tag = f'<img src="{html.escape(img)}">' if img else "🛍️"
                st.markdown(f'<div class="pimg">{tag}</div>', unsafe_allow_html=True)   # fixed-size image box
                st.markdown(f"**{c['rating']} ★** | {c['ratings']} ratings")
                st.markdown(f'<div class="ptitle">{html.escape(c["title"])}</div>', unsafe_allow_html=True)
                st.markdown(f"### ${c['price']:.2f}")
                st.caption(f"{c['category'].replace('_', ' ')} · ID {c['id']}")
                st.markdown(f"[View on Amazon](https://www.amazon.com/dp/{c['id']})")

def render(m):
    with st.chat_message(m["role"]):
        st.write(m["text"])
        if m.get("cards"): show_cards(m["cards"])
        if m.get("orders"): st.dataframe(pd.DataFrame(m["orders"]), hide_index=True)
        if m.get("compare"): st.dataframe(pd.DataFrame(m["compare"]), hide_index=True)
        if m.get("ticket"): st.success(f"Support ticket {m['ticket']} created. A human agent will follow up (demo).")
        if m.get("tools"): st.caption("Tools used: " + ", ".join(dict.fromkeys(m["tools"])))
        if m.get("sources"):
            with st.expander("Sources"): st.write("\n\n".join(m["sources"]))

for m in S.history: render(m)

# ---------- confirmation buttons for actions that change data ----------
if p := S.pending:
    with st.container(border=True):
        st.warning(f"Please confirm: {p['summary']}")
        b1, b2 = st.columns(2)
        if b1.button("✅ Confirm", key="confirm"):
            res = orders.perform_return(p["user"], p["order_id"], p["reason"]) if p["type"] == "return" else orders.perform_cancel(p["user"], p["order_id"])
            add(role="assistant", text=("✅ " if res["ok"] else "⚠️ ") + res["message"])
            S.pending = None; st.rerun()
        if b2.button("Cancel", key="decline"):
            add(role="assistant", text="Okay, I haven't changed anything.")
            S.pending = None; st.rerun()

# ---------- new message ----------
if q := st.chat_input("Ask about products, reviews, returns, orders..."):
    st.chat_message("user").write(q)
    past = [{"role": m["role"], "content": m["text"]} for m in S.history[-8:]]
    add(role="user", text=q)
    try:
        with st.spinner("Working on it..."):
            tb.start(user, q)
            text = run_agent(complete, tb, build_system(user, ", ".join(c.replace("_", " ") for c in config.CATS), S.last_search), past, q)
        c = tb.ctx
        seen, cards = set(), []
        for card in c["cards"]:                       # only show cards the answer actually mentions
            if card["id"] in text and card["id"] not in seen: seen.add(card["id"]); cards.append(card)
        if c["pending"]: S.pending = c["pending"]
        if c["search"]: S.last_search = c["search"]
        add(role="assistant", text=text, cards=cards[:6], orders=c["orders"], compare=c["compare"],
            ticket=c["ticket"], tools=c["tools"], sources=c["sources"])
    except Exception as e:
        add(role="assistant", text=f"Sorry, something went wrong: {e}")
    st.rerun()