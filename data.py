import os, time, requests
from config import CATS, RAW

BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"
FILES = [("review_categories", "{c}"), ("meta_categories", "meta_{c}")]   # (folder, file name)

def fetch(url, path):
    part = path + ".part"
    done = os.path.getsize(part) if os.path.exists(part) else 0
    with requests.get(url, headers={"Range": f"bytes={done}-"} if done else {}, stream=True, timeout=60) as r:
        if r.status_code != 416:                       # 416 = .part is already complete
            r.raise_for_status()
            mode, n = ("ab", done) if r.status_code == 206 else ("wb", 0)
            with open(part, mode) as f:
                for block in r.iter_content(1 << 20):
                    f.write(block); n += len(block)
                    if n // (500 << 20) != (n - len(block)) // (500 << 20):
                        print(f"  {os.path.basename(path)}: {n >> 20} MB")
    os.replace(part, path)
    print("saved", path)

def get(url, path):
    if os.path.exists(path):
        print("have", path); return
    for attempt in range(1, 21):                       # retry on connection resets
        try: return fetch(url, path)
        except requests.exceptions.RequestException as e:
            print(f"  network error ({type(e).__name__}), retry {attempt}/20 in 10s"); time.sleep(10)
    print("FAILED", path)

os.makedirs(RAW, exist_ok=True)
for c in CATS:
    for folder, name in FILES:
        fn = name.format(c=c) + ".jsonl.gz"
        get(f"{BASE}/{folder}/{fn}", f"{RAW}/{fn}")