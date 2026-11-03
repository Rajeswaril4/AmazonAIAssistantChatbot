"""Step 1: shrink raw Amazon files to the best products (+ helpful reviews if available).
Saves each category separately, so a crash or Ctrl+C never loses finished work, then merges."""
import glob, os
import pandas as pd
from config import CATS, RAW, OUT, MAX_PRODUCTS, MAX_REVIEWS

def chunks(name, size=50_000):
    files = sorted(glob.glob(f"{RAW}/{name}.*"), key=lambda f: not f.endswith(".gz"))  # prefer .gz
    return pd.read_json(files[0], lines=True, chunksize=size) if files else None

join = lambda x: " ".join(x) if isinstance(x, list) else ""
def pick(imgs):                      # first usable image URL
    for im in imgs if isinstance(imgs, list) else []:
        im = im or {}
        u = im.get("large") or im.get("hi_res") or im.get("thumb")
        if u: return u
    return ""
os.makedirs(f"{OUT}/cats", exist_ok=True)

for cat in CATS:
    pf, rf = f"{OUT}/cats/{cat}_products.parquet", f"{OUT}/cats/{cat}_reviews.parquet"
    if os.path.exists(pf) and (os.path.exists(rf) or not glob.glob(f"{RAW}/{cat}.*")):
        print(f"{cat}: already done"); continue
    meta = chunks(f"meta_{cat}")
    if meta is None:
        print(f"{cat}: meta file missing - skipped"); continue
    try:
        # products: priced, >= 20 ratings, keep the most-rated
        parts = []
        for ch in meta:
            ch["price"] = pd.to_numeric(ch["price"], errors="coerce")
            parts.append(ch[(ch.rating_number >= 20) & ch.price.notna()])
        p = pd.concat(parts).nlargest(MAX_PRODUCTS, "rating_number")
        p["info"] = (p.features.map(join) + " " + p.description.map(join)).str.strip()
        p["category"] = cat
        p["image"] = p.images.map(pick)
        p[["parent_asin", "title", "price", "average_rating", "rating_number", "info", "category", "image"]].to_parquet(pf)
        print(f"{cat}: {len(p)} products")

        # reviews (optional): only for kept products, most helpful first
        revs = chunks(cat, 200_000)
        if revs is None:
            print("  no review file - products only"); continue
        keep = set(p.parent_asin)
        revs = (ch.rename(columns={"helpful_votes": "helpful_vote"}) for ch in revs)
        r = pd.concat(ch[ch.parent_asin.isin(keep) & (ch.text.str.len() > 30)]
                      [["parent_asin", "rating", "title", "text", "helpful_vote"]] for ch in revs)
        r = r.sort_values("helpful_vote", ascending=False).groupby("parent_asin").head(MAX_REVIEWS)
        r.to_parquet(rf)
        print(f"  {len(r)} reviews")
    except Exception as e:
        print(f"{cat}: FAILED ({e}) - skipped")

# merge finished categories into the two files the index uses
read = lambda kind: [pd.read_parquet(f"{OUT}/cats/{c}_{kind}.parquet")
                     for c in CATS if os.path.exists(f"{OUT}/cats/{c}_{kind}.parquet")]
P, R = read("products"), read("reviews")
pd.concat(P).to_parquet(f"{OUT}/products.parquet")
if R: pd.concat(R).reset_index(drop=True).to_parquet(f"{OUT}/reviews.parquet")
print(f"saved: {sum(map(len, P))} products, {sum(map(len, R))} reviews")