import argparse
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

DEF_SCRAPER = "scraper2.py"
DEF_SEED_FILE = "seeds.txt"
DEF_OUT_DIR = "../data/raw/wikipedia"

def read_seeds(p: Path):
    seeds = []
    if not p.exists():
        print(f"[error] seed file not found: {p}", file=sys.stderr)
        sys.exit(1)
    for line in p.read_text(encoding="utf-8").splitlines():
        u = line.strip()
        if u and not u.startswith("#"):
            seeds.append(u)
    # keep order, drop dups
    seen = set(); uniq = []
    for u in seeds:
        if u not in seen:
            uniq.append(u); seen.add(u)
    return uniq

def build_cmd(scraper_path: Path, seed: str, args):
    cmd = [
        sys.executable, str(scraper_path),
        "--seed", seed,
        "--max-depth", str(args.max_depth),
        "--max-pages", str(args.max_pages),
        "--rate-limit", str(args.rate_limit),
        "--timeout", str(args.timeout),
        "--retries", str(args.retries),
        "--out-dir", str(args.out_dir),
        "--user-agent", args.user_agent,
    ]
    if args.also_linked_pdfs:
        cmd.append("--also-linked-pdfs")
    if args.no_respect_robots:
        cmd.append("--no-respect-robots")
    # default deny to skip obvious noise unless user overrides
    for pat in (args.deny or ["privacy", "terms", "sitemap", "/careers", "/search", "/login"]):
        cmd.extend(["--deny", pat])
    for pat in (args.allow or []):
        cmd.extend(["--allow", pat])
    if not args.same_domain_only:
        cmd.append("--no-same-domain-only")
    return cmd

def main():
    ap = argparse.ArgumentParser(
        description="Fast runner: parallelize scraper2.py across seeds."
    )
    ap.add_argument("--scraper", default=DEF_SCRAPER,
                    help="Path to scraper2.py (default: scraper2.py)")
    ap.add_argument("--seed-file", default=DEF_SEED_FILE,
                    help="Path to seeds.txt (default: seeds.txt)")
    ap.add_argument("--out-dir", default=DEF_OUT_DIR,
                    help="Output dir for cleaned .txt (default: ../data/raw/wikipedia)")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel processes (default: 4)")
    ap.add_argument("--max-depth", type=int, default=3,
                    help="Max crawl depth per seed (default: 3)")
    ap.add_argument("--max-pages", type=int, default=2000,
                    help="Global page cap per process/seed (default: 2000)")
    ap.add_argument("--also-linked-pdfs", action="store_true",
                    help="Also convert PDFs linked on crawled pages")
    ap.add_argument("--rate-limit", type=float, default=0.2,
                    help="Seconds between requests (default: 0.2 – faster)")
    ap.add_argument("--timeout", type=int, default=12,
                    help="HTTP timeout seconds (default: 12 – faster)")
    ap.add_argument("--retries", type=int, default=1,
                    help="HTTP retries (default: 1 – faster)")
    ap.add_argument("--user-agent", default="ANLP-11711 Scraper (+your_andrew@andrew.cmu.edu)",
                    help="Custom User-Agent string")
    ap.add_argument("--deny", action="append",
                    help="Regex to deny (repeatable). If omitted, a sensible deny list is used.")
    ap.add_argument("--allow", action="append",
                    help="Regex to allow (repeatable)")
    ap.add_argument("--same-domain-only", action="store_true", default=True,
                    help="Limit to same-domain HTML pages (default: true)")
    ap.add_argument("--no-same-domain-only", dest="same_domain_only", action="store_false")
    ap.add_argument("--no-respect-robots", action="store_true",
                    help="(Not recommended) ignore robots.txt")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    scraper_path = (root / args.scraper).resolve()
    seed_file = (root / args.seed_file).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = read_seeds(seed_file)
    if not seeds:
        print("[error] no seeds found in seed file.", file=sys.stderr)
        sys.exit(1)

    print(f"[info] seeds: {len(seeds)} | workers: {args.workers} | out: {out_dir}")
    print(f"[info] rate-limit={args.rate_limit}s timeout={args.timeout}s retries={args.retries} depth={args.max_depth}")

    def run_one(seed):
        cmd = build_cmd(scraper_path, seed, args)
        # stream logs so you see progress per seed
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return seed, proc.returncode, proc.stdout

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, s): s for s in seeds}
        for fut in as_completed(futs):
            seed, code, out = fut.result()
            status = "OK" if code == 0 else f"FAIL({code})"
            print(f"\n===== [{status}] {seed} =====")
            # show last ~40 lines from that seed’s scraper run
            tail = "\n".join(out.strip().splitlines()[-40:])
            print(tail)
            results.append((seed, code))

    failures = [s for s, code in results if code != 0]
    if failures:
        print("\n[warn] some seeds failed:")
        for s in failures:
            print(" -", s)
        sys.exit(1)
    else:
        print("\n[done] all seeds completed.")
        sys.exit(0)

if __name__ == "__main__":
    main()
