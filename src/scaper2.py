import argparse, hashlib, io, json, os, re, sys, time
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup
from readability import Document
import pdfplumber
from pypdf import PdfReader
from tqdm import tqdm

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")

def safe_filename_from_url(url: str, ext_hint: str) -> str:
    p = urlparse(url)
    path = p.path or "/"
    fn = path.strip("/").replace("/", "_") or "index"
    if p.query:
        fn += "_" + hashlib.md5(p.query.encode()).hexdigest()[:8]
    if not fn.endswith(ext_hint):
        fn += ext_hint
    return fn

def domain_of(url: str) -> str:
    return urlparse(url).netloc

def is_pdf_url(url: str) -> bool:
    low = url.lower()
    return low.endswith(".pdf") or ".pdf?" in low

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_bytes(path: str, data: bytes):
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        f.write(data)

def save_text(path: str, text: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def html_title(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "lxml")
        if soup.title and soup.title.string:
            return soup.title.string.strip()
    except Exception:
        pass
    return ""

def html_to_clean_text(html: str) -> str:
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

def pdf_to_text(pdf_bytes: bytes) -> str:
    # try pdfplumber first
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
            return "\n".join(pages).strip()
    except Exception:
        pass
    # fallback to pypdf
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for pg in reader.pages:
            pages.append(pg.extract_text() or "")
        return "\n".join(pages).strip()
    except Exception:
        return ""

# ---------- core ----------
def fetch(session: requests.Session, url: str, timeout: int, retries: int = 2):
    last_err = None
    for _ in range(retries + 1):
        try:
            resp = session.get(url, timeout=timeout)
            if 200 <= resp.status_code < 300:
                return resp
        except Exception as e:
            last_err = e
        time.sleep(0.6)
    if last_err:
        raise last_err
    raise RuntimeError(f"Failed to fetch {url}")

def extract_linked_pdfs(html_bytes: bytes, base_url: str) -> list[str]:
    try:
        soup = BeautifulSoup(html_bytes, "lxml")
    except Exception:
        return []
    urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full = urljoin(base_url, href)
        if is_pdf_url(full):
            urls.append(full)
    # de-dup while preserving order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            out.append(u); seen.add(u)
    return out

def process_url(session, url, out_root, ua, also_linked_pdfs, manifest_fp):
    dom = domain_of(url)
    raw_dir = os.path.join(out_root, "raw", SAFE_FILENAME_RE.sub("_", dom))
    txt_dir = os.path.join(out_root, "cleaned", SAFE_FILENAME_RE.sub("_", dom))

    # fetch
    resp = fetch(session, url, timeout=25)
    ctype = (resp.headers.get("Content-Type") or "").lower()
    content = resp.content

    # save raw
    is_pdf = is_pdf_url(url) or ("application/pdf" in ctype)
    raw_leaf = safe_filename_from_url(url, ".pdf" if is_pdf else ".html")
    raw_path = os.path.join(raw_dir, raw_leaf)
    save_bytes(raw_path, content)

    sha = sha256_bytes(content)

    # convert to text
    if is_pdf:
        text = pdf_to_text(content)
        title = ""
        mime = "application/pdf"
    else:
        html = content.decode("utf-8", errors="replace")
        text = html_to_clean_text(html)
        title = html_title(html)
        mime = "text/html"

    # save text
    txt_leaf = safe_filename_from_url(url, ".txt")
    text_path = os.path.join(txt_dir, txt_leaf)
    save_text(text_path, text)

    rec = {
        "url": url,
        "title": title,
        "domain": dom,
        "mime": mime,
        "sha256": sha,
        "raw_path": raw_path,
        "text_path": text_path,
    }
    manifest_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # optionally, pull any PDFs linked ON THIS PAGE (no further crawling)
    new_pdf_urls = []
    if (not is_pdf) and also_linked_pdfs:
        new_pdf_urls = extract_linked_pdfs(content, url)
    return new_pdf_urls

def main():
    ap = argparse.ArgumentParser(description="Minimal: scrape exactly the URLs you list (optional: PDFs linked on those pages).")
    ap.add_argument("--input", required=True, help="Path to a text file with one URL per line")
    ap.add_argument("--out", default="out", help="Output root directory (default: out)")
    ap.add_argument("--also-linked-pdfs", action="store_true", help="Also download PDFs linked from the listed HTML pages")
    ap.add_argument("--user-agent", default=None, help="Custom User-Agent (recommended)")
    args = ap.parse_args()

    ensure_dir(args.out)
    ensure_dir(os.path.join(args.out, "raw"))
    ensure_dir(os.path.join(args.out, "cleaned"))

    urls = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            u = line.strip()
            if u and not u.startswith("#"):
                urls.append(u)

    if not urls:
        print("No URLs found in input file.", file=sys.stderr)
        sys.exit(1)

    ua = args.user_agent or "SimpleScraper/1.0 (+contact@example.com)"
    session = requests.Session()
    session.headers.update({"User-Agent": ua})

    manifest_path = os.path.join(args.out, "manifest.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        pbar = tqdm(urls, desc="Scraping")
        extra_pdf_queue = []
        for url in pbar:
            try:
                new_pdfs = process_url(session, url, args.out, ua, args.also_linked_pdfs, mf)
                extra_pdf_queue.extend(new_pdfs)
            except Exception as e:
                print(f"[warn] {url}: {e}", file=sys.stderr)

        # process one layer of PDFs found on those pages (no recursion)
        if args.also_linked_pdfs and extra_pdf_queue:
            pbar2 = tqdm(extra_pdf_queue, desc="Downloading linked PDFs")
            for pdf_url in pbar2:
                try:
                    process_url(session, pdf_url, args.out, ua, False, mf)
                except Exception as e:
                    print(f"[warn] {pdf_url}: {e}", file=sys.stderr)

    print(f"Done. Manifest: {manifest_path}")

if __name__ == "__main__":
    main()
