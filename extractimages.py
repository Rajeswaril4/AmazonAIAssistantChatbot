"""Extra step (Colab, where data/meta_*.jsonl.gz are): collect one main image URL per product -> images.json.
Run from the same folder as config.py and data/. No index rebuild needed; copy images.json next to app.py."""
import glob, json
import pandas as pd
from config import CATS, RAW, OUT

def main_image(im):
    """Handles both raw-jsonl format (list of dicts) and Hugging Face format (dict of lists)."""
    if isinstance(im, dict):
        for k in ("large", "hi_res", "thumb"):
            for u in im.get(k) or []:
                if u: return u
        return ""
    if isinstance(im, list) and im:
        first = next((i for i in im if isinstance(i, dict) and i.get("variant") == "MAIN"), im[0])
        if isinstance(first, dict):
            return first.get("large") or first.get("hi_res") or first.get("thumb") or ""
    return ""

if __name__ == "__main__":
    keep = set(pd.read_parquet(f"{OUT}/products.parquet", columns=["parent_asin"]).parent_asin)
    images = {}
    for cat in CATS:
        files = sorted(glob.glob(f"{RAW}/meta_{cat}.*"), key=lambda f: not f.endswith(".gz"))
        if not files:
            print(f"{cat}: meta file missing - skipped"); continue
        for ch in pd.read_json(files[0], lines=True, chunksize=50_000):
            ch = ch[ch.parent_asin.isin(keep)]
            for asin, im in zip(ch.parent_asin, ch.images):
                if u := main_image(im): images[asin] = u
        print(f"{cat}: total images so far {len(images)}")
    json.dump(images, open("images.json", "w"))
    print(f"saved images.json: {len(images)} of {len(keep)} products have an image")