"""
Eval

This file is an evaluation script
for comparing an LLM generated output
against the reference answers.

Computes relevant metrics such as
exact match or F1.

Now supports grouping question ranges for per-group analysis.
"""

import json, re, string
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Union, Tuple

_ARTICLES = {"a", "an", "the"}
_PUNC_TABLE = str.maketrans({c: " " for c in string.punctuation})

_WRAPPER_PATTERNS = [
    r"^\s*(final answer|the answer is|answer)\s*[:\-]\s*",
    r"\s*(={2,}.*={2,}|end answer|</s>)\s*$",
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
    s = s.translate(_PUNC_TABLE)
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
    rec = match / len(g_toks)
    return 2 * prec * rec / (prec + rec)


def exact_match_score(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def _ensure_list(v: Union[str, List[str]]) -> List[str]:
    if isinstance(v, list):
        return v
    return [v]


def load_answers(path: Union[str, Path]) -> Dict[str, List[str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(k): _ensure_list(v) for k, v in data.items()}


def evaluate(
    gold_path: str,
    pred_path: str,
    groups: List[Tuple[int, int]] = None,
):
    """
    Evaluates EM/F1 for the dataset.

    If `groups` is provided, returns per-group metrics as well.
    """
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

    # --- Compute per-group summaries if groups are provided ---
    group_summaries = {}
    if groups:
        for start, end in groups:
            group_ids = [str(i) for i in range(start, end + 1) if str(i) in gold]
            if not group_ids:
                continue

            em_sum, f1_sum = 0.0, 0.0
            for qid in group_ids:
                item = next(x for x in per_item if x["id"] == qid)
                em_sum += item["best_em"]
                f1_sum += item["best_f1"]
            n = len(group_ids)
            group_summaries[f"{start}-{end}"] = {
                "count": n,
                "EM": round(100.0 * em_sum / n, 2),
                "F1": round(100.0 * f1_sum / n, 2),
            }

    return summary, per_item, group_summaries


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="SQuAD-style EM/F1 evaluator (with group support)")
    ap.add_argument("--gold", required=True, help="path to reference_answers_*.json")
    ap.add_argument("--pred", required=True, help="path to system_output.json")
    ap.add_argument("--dump-per-item", default=None, help="optional path to save per-question scores JSON")
    ap.add_argument(
        "--groups",
        nargs="+",
        default=None,
        help="optional list of start:end pairs, e.g. 1:10 11:25 26:50",
    )
    args = ap.parse_args()

    # Parse --groups argument
    groups = None
    if args.groups:
        groups = []
        for g in args.groups:
            start, end = map(int, g.split(":"))
            groups.append((start, end))

    summary, per_item, group_summaries = evaluate(args.gold, args.pred, groups)

    print("== Summary ==")
    print(json.dumps(summary, indent=2))

    if groups:
        print("\n== Group Summaries ==")
        print(json.dumps(group_summaries, indent=2))

    if args.dump_per_item:
        Path(args.dump_per_item).write_text(json.dumps(per_item, indent=2), encoding="utf-8")
        print(f"Per-item details written to {args.dump_per_item}")

# groups:
# 1:14 15:33 34:42 43:55 56:80 81:99 100:103 104:106 107:144 145:150