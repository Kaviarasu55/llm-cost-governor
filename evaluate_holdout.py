"""
evaluate_holdout.py — Intelligent LLM Cost Governor, Phase 3

Head-to-head comparison of the rule-based classifier (classifier.py)
and the ML classifier (ml_classifier.py) on holdout_test_set.csv —
a separately generated, naturally/messily-phrased set the models were
NOT trained or tuned on. This is the number that goes in the README,
not the training-set cross-validation score.

Run:
    python evaluate_holdout.py
"""

import csv
from collections import defaultdict

from sklearn.metrics import classification_report

import classifier
import ml_classifier

HOLDOUT_PATH = "holdout_test_set.csv"


def load_holdout(path: str):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def evaluate(name: str, predict_fn, rows):
    y_true, y_pred = [], []
    mismatches = []

    for row in rows:
        query = row["query"]
        true_type = row["type"]
        pred_type, _ = predict_fn(query)

        y_true.append(true_type)
        y_pred.append(pred_type)

        if pred_type != true_type:
            mismatches.append((query, true_type, pred_type))

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(rows)

    print(f"\n{'=' * 60}")
    print(f"{name} — accuracy on holdout: {correct}/{len(rows)} = {accuracy:.1%}")
    print(f"{'=' * 60}")
    print(classification_report(y_true, y_pred, zero_division=0))

    print(f"Mismatches ({len(mismatches)}):")
    for query, true_t, pred_t in mismatches:
        print(f"  TRUE={true_t:20s} PRED={pred_t:20s} | {query[:70]}")

    return accuracy, mismatches


def main():
    rows = load_holdout(HOLDOUT_PATH)
    print(f"Loaded {len(rows)} holdout examples.")

    rule_acc, rule_mismatches = evaluate(
        "RULE-BASED (classifier.py)", classifier.classify_query, rows
    )
    ml_acc, ml_mismatches = evaluate(
        "ML (ml_classifier.py)", ml_classifier.classify_query_ml, rows
    )

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Rule-based accuracy: {rule_acc:.1%}")
    print(f"ML accuracy:         {ml_acc:.1%}")
    print(f"Difference:          {(ml_acc - rule_acc) * 100:+.1f} points")


if __name__ == "__main__":
    main()