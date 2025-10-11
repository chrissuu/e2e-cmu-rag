# FILE: src/eval_with_significance.py
import json, re, string
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Union
import numpy as np
from scipy.stats import ttest_rel

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
    rec  = match / len(g_toks)
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

def evaluate_single(gold_path: str, pred_path: str):
    gold = load_answers(gold_path)
    pred_raw = json.loads(Path(pred_path).read_text(encoding="utf-8"))
    preds = {str(k): str(v) for k, v in pred_raw.items()}

    ids = sorted(gold.keys(), key=lambda x: int(x) if x.isdigit() else x)
    total = len(ids)
    em_scores, f1_scores = [], []
    per_item = []

    for qid in ids:
        gold_list = gold[qid]
        p = preds.get(qid, "")
        em = max(exact_match_score(p, g) for g in gold_list)
        f1 = max(f1_score(p, g) for g in gold_list)
        em_scores.append(1.0 if em else 0.0)
        f1_scores.append(f1)
        per_item.append({
            "id": qid,
            "prediction": p,
            "best_em": int(em),
            "best_f1": round(f1, 4),
            "gold": gold_list,
        })

    summary = {
        "count": total,
        "EM": round(100.0 * np.mean(em_scores), 2),
        "F1": round(100.0 * np.mean(f1_scores), 2),
    }
    return summary, per_item, np.array(em_scores), np.array(f1_scores)

def significance_test(scores_a: np.ndarray, scores_b: np.ndarray, metric_name="Metric"):
    """
    Paired t-test between two arrays of per-item scores
    """
    t_stat, p_value = ttest_rel(scores_a, scores_b)
    mean_diff = np.mean(scores_a - scores_b)
    print(f"\nSignificance test for {metric_name}:")
    print(f"Mean difference (A - B): {mean_diff:.4f}")
    print(f"Paired t-test p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("=> Statistically significant difference at p<0.05")
    else:
        print("=> No significant difference")

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="SQuAD-style EM/F1 evaluator with significance testing")
    ap.add_argument("--gold", required=True, help="path to reference_answers.json")
    ap.add_argument("--preds", nargs=2, required=True, help="paths to two system output JSON files")
    ap.add_argument("--dump-per-item", default=None, help="optional path to save per-question scores JSON")
    args = ap.parse_args()

    summary_a, per_item_a, em_a, f1_a = evaluate_single(args.gold, args.preds[0])
    summary_b, per_item_b, em_b, f1_b = evaluate_single(args.gold, args.preds[1])

    print("== Summary Method A ==")
    print(json.dumps(summary_a, indent=2))
    print("== Summary Method B ==")
    print(json.dumps(summary_b, indent=2))

    significance_test(em_a, em_b, metric_name="Exact Match")
    significance_test(f1_a, f1_b, metric_name="F1 Score")

    if args.dump_per_item:
        combined_per_item = [{"id": x["id"],
                              "pred_a": x["prediction"],
                              "pred_b": y["prediction"],
                              "best_em_a": x["best_em"],
                              "best_em_b": y["best_em"],
                              "best_f1_a": x["best_f1"],
                              "best_f1_b": y["best_f1"],
                              "gold": x["gold"]}
                             for x, y in zip(per_item_a, per_item_b)]
        Path(args.dump_per_item).write_text(json.dumps(combined_per_item, indent=2), encoding="utf-8")
        print(f"Per-item details written to {args.dump_per_item}")
