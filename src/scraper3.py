# FILE: src/scraper3.py
import argparse
import hashlib
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from readability import Document
import pdfplumber
from pypdf import PdfReader


# --------------------------
# Defaults / paths
# --------------------------
DEFAULT_SEED_FILE = "seeds.txt"                 # relative to this script (src/)
DEFAULT_OUT_DIR   = (Path(__file__).parent / ".." / "data" / "raw2").resolve()
DEFAULT_UA        = "ANLP-11711 StaticScraper (+contact@example.com)"


# --------------------------
# Utilities
# --------------------------
SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def is_pdf_url(url: str) -> bool:
    base = url.split("?", 1)[0].lower()
    return base.endswith(".pdf")

def safe_leaf_from_url(url: str, ext_hint: str) -> str:
    """
    Build a filesystem-safe leaf filename from a URL, with a deterministic suffix
    if the URL has a query string. Force a particular extension via ext_hint.
    """
    p = urlparse(url)
    path = p.path or "/"
    leaf = path.strip("/").replace("/", "_") or "index"
    if p.query:
        leaf += "_" + hashlib.md5(p.query.encode()).hexdigest()[:8]
    if not leaf.endswith(ext_hint):
        leaf += ext_hint
    # sanitize
    leaf = SAFE_FILENAME_RE.sub("_", leaf)
    return leaf

def polite_get(session: requests.Session, url: str, timeout: int, retries: int, rate_limit_s: float):
    last_err = None
    for attempt in range(retries + 1):
        try:
            if rate_limit_s > 0:
                time.sleep(rate_limit_s)
            resp = session.get(url, timeout=timeout, allow_redirects=True)
            if 200 <= resp.status_code < 300:
                return resp
            last_err = RuntimeError(f"HTTP {resp.status_code}")
        except Exception as e:
            last_err = e
        # simple backoff
        time.sleep(0.5 * (attempt + 1))
    raise last_err

def html_to_clean_text(html: str) -> str:
    """
    Prefer readability summary; fall back to a basic strip of visible text.
    """
    try:
        doc = Document(html)
        content_html = doc.summary()
        soup = BeautifulSoup(content_html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    for sel in ["nav", "header", "footer", "aside"]:
        for t in soup.select(sel):
            t.decompose()

    lines = []
    for el in soup.find_all(["h1","h2","h3","h4","h5","h6","p","li","td","th"]):
        txt = el.get_text(" ", strip=True)
        if txt:
            lines.append(txt)
    return "\n".join(lines).strip()

def pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    # try pdfplumber first
    try:
        import io
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
            return "\n".join(pages).strip()
    except Exception:
        pass
    # fallback to pypdf
    try:
        import io
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for pg in reader.pages:
            pages.append(pg.extract_text() or "")
        return "\n".join(pages).strip()
    except Exception:
        return ""


# --------------------------
# Core: process one URL
# --------------------------
def process_url(url: str, out_root: Path, session: requests.Session, timeout: int, retries: int, rate_limit_s: float):
    """
    Fetch a single seed URL and save:
      - If HTML: raw HTML to out_root/html/*.html and cleaned text to out_root/text/*.txt
      - If PDF:  raw PDF  to out_root/pdf/*.pdf  and extracted text to out_root/text/*.txt
    """
    resp = polite_get(session, url, timeout=timeout, retries=retries, rate_limit_s=rate_limit_s)
    content_type = (resp.headers.get("Content-Type") or "").lower()

    html_dir = out_root / "html"
    text_dir = out_root / "text"
    pdf_dir  = out_root / "pdf"
    ensure_dir(html_dir); ensure_dir(text_dir); ensure_dir(pdf_dir)

    if is_pdf_url(url) or "application/pdf" in content_type:
        leaf_pdf  = safe_leaf_from_url(url, ".pdf")
        pdf_path  = pdf_dir / leaf_pdf
        text_path = text_dir / safe_leaf_from_url(url, ".txt")

        with open(pdf_path, "wb") as f:
            f.write(resp.content)

        extracted = pdf_bytes_to_text(resp.content)
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(extracted or "")

        return {"url": url, "type": "pdf", "pdf_path": str(pdf_path), "text_path": str(text_path)}

    # Otherwise, treat as HTML
    try:
        html = resp.content.decode("utf-8", errors="replace")
    except Exception:
        html = resp.text  # last resort

    leaf_html = safe_leaf_from_url(url, ".html")
    leaf_txt  = safe_leaf_from_url(url, ".txt")

    html_path = html_dir / leaf_html
    text_path = text_dir / leaf_txt

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    cleaned = html_to_clean_text(html)
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(cleaned or "")

    return {"url": url, "type": "html", "html_path": str(html_path), "text_path": str(text_path)}


# --------------------------
# Seeds
# --------------------------
def read_seeds(seed_file: Path):
    if not seed_file.exists():
        print(f"[error] seed file not found: {seed_file}", file=sys.stderr)
        sys.exit(1)
    seeds = []
    for line in seed_file.read_text(encoding="utf-8").splitlines():
        u = line.strip()
        if u and not u.startswith("#"):
            seeds.append(u)
    # de-dupe, preserve order
    seen, uniq = set(), []
    for u in seeds:
        if u not in seen:
            uniq.append(u); seen.add(u)
    return uniq


# --------------------------
# CLI
# --------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Static page scraper: fetch EXACT URLs in seeds.txt, no crawling. Saves raw HTML/PDF and cleaned text to data/raw2."
    )
    ap.add_argument("--seed-file", default=DEFAULT_SEED_FILE,
                    help="Path to seeds file (default: seeds.txt in src/)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                    help="Output root directory (default: ../data/raw2 from src/)")
    ap.add_argument("--user-agent", default=DEFAULT_UA,
                    help="Custom User-Agent")
    ap.add_argument("--timeout", type=int, default=25,
                    help="HTTP timeout seconds (default: 25)")
    ap.add_argument("--retries", type=int, default=2,
                    help="HTTP retries (default: 2)")
    ap.add_argument("--rate-limit", type=float, default=0.0,
                    help="Seconds between requests (default: 0.0)")
    args = ap.parse_args()

    # Seed file path is relative to this script unless absolute
    seed_path = Path(args.seed_file)
    if not seed_path.is_absolute():
        seed_path = (Path(__file__).parent / seed_path).resolve()

    out_root = Path(args.out_dir).resolve()
    ensure_dir(out_root)

    seeds = read_seeds(seed_path)
    if not seeds:
        print("[error] no seeds found in seed file.", file=sys.stderr)
        sys.exit(1)

    session = requests.Session()
    session.headers.update({"User-Agent": args.user_agent})

    print(f"[info] seeds: {len(seeds)} | out: {out_root}")
    print(f"[info] timeout={args.timeout}s retries={args.retries} rate-limit={args.rate_limit}s")

    successes, failures = 0, 0
    for i, url in enumerate(seeds, 1):
        try:
            info = process_url(
                url=url,
                out_root=out_root,
                session=session,
                timeout=args.timeout,
                retries=args.retries,
                rate_limit_s=args.rate_limit
            )
            kind = info.get("type")
            if kind == "pdf":
                print(f"[{i}/{len(seeds)}] PDF  saved: {info['pdf_path']} | text: {info['text_path']}")
            else:
                print(f"[{i}/{len(seeds)}] HTML saved: {info['html_path']} | text: {info['text_path']}")
            successes += 1
        except Exception as e:
            print(f"[{i}/{len(seeds)}] FAIL {url} :: {e}", file=sys.stderr)
            failures += 1

    print(f"\n[done] total={len(seeds)} ok={successes} fail={failures}")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
