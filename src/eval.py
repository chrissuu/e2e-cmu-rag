# FILE: src/eval.py
import json, re, string
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Union

_ARTICLES = {"a", "an", "the"}
_PUNC_TABLE = str.maketrans({c: " " for c in string.punctuation})

_WRAPPER_PATTERNS = [
    r"^\s*(final answer|the answer is|answer)\s*[:\-]\s*",  # leading wrappers
    r"\s*(={2,}.*={2,}|end answer|</s>)\s*$",               # trailing banners/tokens
]
_WRAPPER_RE = [re.compile(p, flags=re.I) for p in _WRAPPER_PATTERNS]

def strip_wrappers(s: str) -> str:
    t = s.strip()
    for rx in _WRAPPER_RE:
        t = rx.sub("", t)
    return t.strip()

def normalize_answer(s: str) -> str:
    s = strip_wrappers(s)
    s = s.lower()
    s = s.translate(_PUNC_TABLE)               # remove punctuation
    tokens = [w for w in s.split() if w not in _ARTICLES]
    return " ".join(tokens)

def f1_score(pred: str, gold: str) -> float:
    p_toks = normalize_answer(pred).split()
    g_toks = normalize_answer(gold).split()
    if not p_toks and not g_toks:
        return 1.0
    if not p_toks or not g_toks:
        return 0.0
    common = defaultdict(int)
    for w in g_toks:
        common[w] += 1
    match = 0
    for w in p_toks:
        if common[w] > 0:
            match += 1
            common[w] -= 1
    if match == 0:
        return 0.0
    prec = match / len(p_toks)
    rec  = match / len(g_toks)
    return 2 * prec * rec / (prec + rec)

def exact_match_score(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)

def _ensure_list(v: Union[str, List[str]]) -> List[str]:
    if isinstance(v, list):
        return v
    return [v]

def load_answers(path: Union[str, Path]) -> Dict[str, List[str]]:
    """
    Accepts either:
      { "1": "upmc", "2": "schenley park", ... }
      or
      { "1": ["upmc", "univ of pittsburgh medical center"], ... }
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(k): _ensure_list(v) for k, v in data.items()}

def evaluate(gold_path: str, pred_path: str):
    gold = load_answers(gold_path)
    pred_raw = json.loads(Path(pred_path).read_text(encoding="utf-8"))
    preds = {str(k): str(v) for k, v in pred_raw.items()}

    ids = sorted(gold.keys(), key=lambda x: int(x) if x.isdigit() else x)
    total = len(ids)
    em_total, f1_total = 0.0, 0.0
    per_item = []

    for qid in ids:
        gold_list = gold[qid]
        p = preds.get(qid, "")
        # SQuAD convention: score = max over all gold variants
        em = max(exact_match_score(p, g) for g in gold_list)
        f1 = max(f1_score(p, g) for g in gold_list)
        em_total += 1.0 if em else 0.0
        f1_total += f1
        per_item.append({
            "id": qid,
            "prediction": p,
            "best_em": int(em),
            "best_f1": round(f1, 4),
            "gold": gold_list,
        })

    summary = {
        "count": total,
        "EM": round(100.0 * em_total / total, 2),
        "F1": round(100.0 * f1_total / total, 2),
    }
    return summary, per_item

if __name__ == "__main__":
    import argparse, pprint
    ap = argparse.ArgumentParser(description="SQuAD-style EM/F1 evaluator")
    ap.add_argument("--gold", required=True, help="path to reference_answers_*.json")
    ap.add_argument("--pred", required=True, help="path to system_output.json")
    ap.add_argument("--dump-per-item", default=None, help="optional path to save per-question scores JSON")
    args = ap.parse_args()

    summary, per_item = evaluate(args.gold, args.pred)
    print("== Summary ==")
    print(json.dumps(summary, indent=2))
    if args.dump_per_item:
        Path(args.dump_per_item).write_text(json.dumps(per_item, indent=2), encoding="utf-8")
        print(f"Per-item details written to {args.dump_per_item}")
