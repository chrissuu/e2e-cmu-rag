import argparse, hashlib, io, json, os, queue, re, sys, time
from dataclasses import dataclass
from urllib.parse import urlparse, urljoin
from urllib import robotparser

import requests
from bs4 import BeautifulSoup
from readability import Document
import pdfplumber
from pypdf import PdfReader
from tqdm import tqdm

# ---------- utilities ----------
SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")

def domain_of(url: str) -> str:
    return urlparse(url).netloc

def same_domain(a: str, b: str) -> bool:
    return domain_of(a).lower() == domain_of(b).lower()

def normalize(url: str) -> str:
    p = urlparse(url)
    fragless = p._replace(fragment="")
    path = re.sub(r"/{2,}", "/", fragless.path or "/")
    netloc = fragless.netloc.replace(":80", "").replace(":443", "")
    return fragless._replace(path=path, netloc=netloc).geturl()

def is_pdf_url(url: str) -> bool:
    base = url.split("?", 1)[0].lower()
    return base.endswith(".pdf")

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def safe_leaf_from_url(url: str, ext_hint: str) -> str:
    p = urlparse(url)
    path = p.path or "/"
    fn = path.strip("/").replace("/", "_") or "index"
    if p.query:
        fn += "_" + hashlib.md5(p.query.encode()).hexdigest()[:8]
    if not fn.endswith(ext_hint):
        fn += ext_hint
    return fn

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

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
    # try readability → fall back to simple stripping
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
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
            return "\n".join(pages).strip()
    except Exception:
        pass
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for pg in reader.pages:
            pages.append(pg.extract_text() or "")
        return "\n".join(pages).strip()
    except Exception:
        return ""

# ---------- fetching ----------
def build_session(user_agent: str):
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent})
    return s

def polite_get(session, url, timeout, retries, rate_limit_s):
    last_err = None
    for attempt in range(retries + 1):
        try:
            time.sleep(max(0.0, rate_limit_s))
            resp = session.get(url, timeout=timeout)
            if 200 <= resp.status_code < 300:
                return resp
            last_err = RuntimeError(f"HTTP {resp.status_code}")
        except Exception as e:
            last_err = e
        time.sleep(0.6 * (attempt + 1))
    raise last_err

# ---------- crawl config ----------
@dataclass
class CrawlOpts:
    seeds: list
    max_depth: int
    max_pages: int
    allow_regex: list
    deny_regex: list
    also_linked_pdfs: bool
    same_domain_only: bool
    out_root: str
    user_agent: str
    timeout: int
    retries: int
    rate_limit_s: float
    respect_robots: bool

def compile_res(pats):
    return [re.compile(p) for p in pats]

def allowed(url: str, allow_res, deny_res):
    if any(r.search(url) for r in deny_res):
        return False
    if not allow_res:
        return True
    return any(r.search(url) for r in allow_res)

def get_robot_parser(seed_url: str, respect: bool):
    rp = robotparser.RobotFileParser()
    if not respect:
        rp.can_fetch = lambda *args, **kwargs: True
        return rp
    robots_url = f"{urlparse(seed_url).scheme}://{domain_of(seed_url)}/robots.txt"
    try:
        rp.set_url(robots_url); rp.read()
    except Exception:
        rp.can_fetch = lambda *args, **kwargs: True
    return rp

# ---------- main crawl ----------
def crawl(opts: CrawlOpts):
    session = build_session(opts.user_agent)
    seen = set()
    q = queue.Queue()
    total_saved = 0

    manifest_path = os.path.join(opts.out_root, "manifest.jsonl")
    ensure_dir(opts.out_root)
    ensure_dir(os.path.join(opts.out_root, "raw"))
    ensure_dir(os.path.join(opts.out_root, "cleaned"))

    rp_map = {}
    for s in opts.seeds:
        d = domain_of(s)
        if d not in rp_map:
            rp_map[d] = get_robot_parser(s, opts.respect_robots)

    allow_res = compile_res(opts.allow_regex)
    deny_res  = compile_res(opts.deny_regex)

    for s in opts.seeds:
        q.put((normalize(s), 0))

    with open(manifest_path, "w", encoding="utf-8") as mf:
        pbar = tqdm(total=opts.max_pages, desc="Crawling")
        while not q.empty() and total_saved < opts.max_pages:
            url, depth = q.get()
            if url in seen:
                continue
            seen.add(url)

            rp = rp_map.get(domain_of(url))
            if rp is None:
                rp = get_robot_parser(url, opts.respect_robots)
                rp_map[domain_of(url)] = rp
            if not rp.can_fetch(opts.user_agent, url):
                continue

            try:
                resp = polite_get(session, url, opts.timeout, opts.retries, opts.rate_limit_s)
            except Exception as e:
                print(f"[warn] fetch failed: {url} :: {e}", file=sys.stderr)
                continue

            ctype = (resp.headers.get("Content-Type") or "").lower()
            content = resp.content
            dom = domain_of(url)
            dom_dir = SAFE_FILENAME_RE.sub("_", dom)

            is_pdf = is_pdf_url(url) or ("application/pdf" in ctype)
            raw_leaf = safe_leaf_from_url(url, ".pdf" if is_pdf else ".html")
            raw_path = os.path.join(opts.out_root, "raw", dom_dir, raw_leaf)
            save_bytes(raw_path, content)

            if is_pdf:
                txt = pdf_to_text(content)
                title = ""
                mime = "application/pdf"
            else:
                html = content.decode("utf-8", errors="replace")
                txt  = html_to_clean_text(html)
                title = html_title(html)
                mime = "text/html"

            txt_leaf = safe_leaf_from_url(url, ".txt")
            text_path = os.path.join(opts.out_root, "cleaned", dom_dir, txt_leaf)
            save_text(text_path, txt)

            rec = {
                "url": url,
                "title": title,
                "domain": dom,
                "mime": mime,
                "sha256": sha256_bytes(content),
                "raw_path": raw_path,
                "text_path": text_path,
                "depth": depth,
            }
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total_saved += 1
            pbar.update(1)

            # enqueue links if html and depth allows
            if (not is_pdf) and (depth < opts.max_depth):
                try:
                    soup = BeautifulSoup(content, "lxml")
                except Exception:
                    soup = None
                if soup:
                    base = url
                    for a in soup.find_all("a", href=True):
                        nxt = urljoin(base, a["href"].strip())
                        nxt = normalize(nxt)

                        # PDFs: optional off-domain inclusion
                        if is_pdf_url(nxt):
                            if opts.also_linked_pdfs and nxt not in seen:
                                q.put((nxt, depth + 1))
                            continue

                        # HTML pages: same-domain by default
                        if opts.same_domain_only and not same_domain(url, nxt):
                            continue

                        if allowed(nxt, allow_res, deny_res):
                            if nxt not in seen:
                                q.put((nxt, depth + 1))
        pbar.close()

    print(f"Done. Saved {total_saved} pages. Manifest: {manifest_path}")
    print(f"Text files ready at: {os.path.join(opts.out_root, 'cleaned')}")

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Auto-scrape seeds and their same-domain subpages (clean text output).")
    ap.add_argument("--seed", action="append", help="Seed URL (repeatable).")
    ap.add_argument("--seed-file", help="Path to a text file with one seed URL per line.")
    ap.add_argument("--max-depth", type=int, default=3, help="Max link depth from seeds (default: 3).")
    ap.add_argument("--max-pages", type=int, default=2000, help="Global page cap (default: 2000).")
    ap.add_argument("--also-linked-pdfs", action="store_true", help="Also download PDFs linked on crawled pages.")
    ap.add_argument("--same-domain-only", action="store_true", default=True, help="Limit to same-domain HTML pages (default).")
    ap.add_argument("--no-same-domain-only", dest="same_domain_only", action="store_false", help="Allow cross-domain HTML pages if allowed by regex.")
    ap.add_argument("--allow", action="append", default=[], help="Regex of URLs to allow (repeatable).")
    ap.add_argument("--deny", action="append", default=[], help="Regex of URLs to deny (repeatable).")
    ap.add_argument("--out", default="out", help="Output root directory (default: out)")
    ap.add_argument("--user-agent", default="AutoScraper/1.0 (+contact@example.com)", help="Custom User-Agent (add a contact email).")
    ap.add_argument("--timeout", type=int, default=25, help="HTTP timeout seconds (default: 25)")
    ap.add_argument("--retries", type=int, default=2, help="HTTP retries (default: 2)")
    ap.add_argument("--rate-limit", type=float, default=1.0, help="Seconds between requests (default: 1.0)")
    ap.add_argument("--respect-robots", action="store_true", default=True, help="Respect robots.txt (default).")
    ap.add_argument("--no-respect-robots", dest="respect_robots", action="store_false", help="Do not respect robots.txt (use with caution).")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    seeds = list(args.seed or [])
    if args.seed_file:
        with open(args.seed_file, "r", encoding="utf-8") as f:
            for line in f:
                u = line.strip()
                if u and not u.startswith("#"):
                    seeds.append(u)
    if not seeds:
        print("No seeds provided. Use --seed or --seed-file.", file=sys.stderr)
        sys.exit(1)

    opts = CrawlOpts(
        seeds=seeds,
        max_depth=args.max_depth,
        max_pages=args.max_pages,
        allow_regex=args.allow,
        deny_regex=args.deny,
        also_linked_pdfs=args.also_linked_pdfs,
        same_domain_only=args.same_domain_only,
        out_root=args.out,
        user_agent=args.user_agent,
        timeout=args.timeout,
        retries=args.retries,
        rate_limit_s=args.rate_limit,
        respect_robots=args.respect_robots,
    )
    crawl(opts)
