"""Step 2 (Colab GPU): embed policies, products and reviews into ChromaDB."""
import glob, os, re
import chromadb, pandas as pd
from sentence_transformers import SentenceTransformer
from config import OUT, POLICY, DB, EMBED_MODEL

model = SentenceTransformer(EMBED_MODEL)
db = chromadb.PersistentClient(path=DB)
embed = lambda texts: model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=True).tolist()

def collection(name):
    try: db.delete_collection(name)          # rebuild from scratch
    except Exception: pass
    return db.create_collection(name, metadata={"hnsw:space": "cosine"})

def add(col, ids, docs, metas, step=2000):
    for i in range(0, len(docs), step):
        s = slice(i, i + step)
        col.add(ids=ids[s], documents=docs[s], embeddings=embed(docs[s]), metadatas=metas[s])

def chunk_md(text, size=1200):
    """Split on blank lines (paragraphs), never mid-sentence. Returns [(section_heading, chunk)]."""
    out, head, buf, buf_head, last = [], "", "", "", ""
    def pieces():                                   # paragraphs; oversized ones are split on sentences
        for b in filter(None, (b.strip() for b in re.split(r"\n\s*\n", text))):
            if len(b) <= size: yield b; continue
            cur = ""
            for s in re.split(r"(?<=[.!?;:])\s+|\n(?=[-*\d])", b):
                if cur and len(cur) + len(s) > size: yield cur; cur = ""
                cur += (" " if cur else "") + s
            if cur: yield cur
    for block in pieces():
        if block.startswith("#"): head = block.lstrip("# ").split("\n")[0][:100]
        if buf and len(buf) + len(block) > size:
            out.append((buf_head, buf))
            buf, buf_head = (last if len(last) < 300 else ""), head      # small overlap: repeat last paragraph
        if not buf: buf_head = head
        buf += ("\n\n" if buf else "") + block; last = block
    if buf: out.append((buf_head, buf))
    return [(h, c) for h, c in out if len(c) >= 40]     # drop empty/tiny fragments

# policies: first line of each file is "source: <url>"
ids, docs, metas = [], [], []
for f in sorted(glob.glob(f"{POLICY}/*.md")):
    lines = open(f, encoding="utf-8").read().split("\n")
    src = lines[0].replace("source:", "").strip() or f
    for i, (head, chunk) in enumerate(chunk_md("\n".join(lines[1:]))):
        ids.append(f"{os.path.basename(f)}-{i}")
        docs.append(f"{head}\n{chunk}" if head and not chunk.startswith("#") else chunk)
        metas.append({"source": src, "section": head, "file": os.path.basename(f)})
print(f"policy chunks: {len(docs)}")
if docs: add(collection("policy"), ids, docs, metas)

# products: searchable by meaning, filterable by price
p = pd.read_parquet(f"{OUT}/products.parquet").reset_index(drop=True)
print(f"products: {len(p)}")
add(collection("products"), p.parent_asin.tolist(),
    (p.title + ". " + p["info"].str[:800]).tolist(),
    [dict(title=t, price=float(pr), rating=float(a), n=int(n), category=c, image=im or "")
     for t, pr, a, n, c, im in zip(p.title, p.price, p.average_rating, p.rating_number, p.category,
                                   p["image"] if "image" in p else [""] * len(p))])

# reviews (only if prepare_data found review files)
if os.path.exists(f"{OUT}/reviews.parquet"):
    r = pd.read_parquet(f"{OUT}/reviews.parquet").reset_index(drop=True)
    print(f"reviews: {len(r)}")
    add(collection("reviews"), r.index.astype(str).tolist(),
        (r.title.fillna("") + ". " + r.text.str[:1000]).tolist(),
        [dict(asin=a, rating=float(x)) for a, x in zip(r.parent_asin, r.rating)])
print("done")