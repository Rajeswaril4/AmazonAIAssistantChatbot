"""Download the policy pages listed in policy_urls.txt ("name | url" per line) into data/amazon/*.md.
Needs: pip install trafilatura requests. Blocked pages are reported - copy those by hand."""
import os, time
import requests, trafilatura
from config import POLICY

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
           "Accept-Language": "en-US,en;q=0.9"}

def get_text(url):
    html = trafilatura.fetch_url(url)                       # try 1: trafilatura
    if not html:
        try: html = requests.get(url, headers=HEADERS, timeout=30).text   # try 2: browser-like request
        except requests.RequestException: return None
    text = trafilatura.extract(html, output_format="markdown") if html else None
    bad = ("captcha", "enter the characters you see", "sign in")
    return text if text and len(text) > 500 and not any(b in text.lower()[:600] for b in bad) else None

os.makedirs(POLICY, exist_ok=True)
for line in open("policy_urls.txt", encoding="utf-8"):
    if "|" not in line: continue
    name, url = (s.strip() for s in line.split("|", 1))
    text = get_text(url)
    if text:
        open(f"{POLICY}/{name}.md", "w", encoding="utf-8").write(f"source: {url}\n\n{text}")
        print(f"saved   {name} ({len(text)} chars)")
    else:
        print(f"BLOCKED {name} -> copy by hand from {url}")
    time.sleep(2)                                           # be polite