"""
Threshold calibration for the class-weighted DistilBERT model (README section 5.4).

Default classification uses argmax (== a 0.50 cutoff on P(Special)). Here we
instead choose the decision threshold on the VALIDATION set and only then report
on the locked TEST set, so the operating point is not tuned on test data.

Loads the checkpoint trained by finetune_weighted.py (no retraining).
"""
import torch  # MUST precede numpy/pandas on this box (c10.dll WinError 1114)
import json
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support, accuracy_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SEED = 42
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 96
LABEL_MAP = {"0": 0, "1": 1}
SPECIAL = 1
ckpt = sorted(glob.glob("takemeter-model-weighted/checkpoint-*"))[-1]
print("Loading checkpoint:", ckpt)

# ── Reproduce the exact split ─────────────────────────────────────────────
df = pd.read_csv("./data/menu_item.csv")
df["label"] = df["label"].astype(str)
df["label_id"] = df["label"].map(LABEL_MAP)
df = df.dropna(subset=["label_id"]); df["label_id"] = df["label_id"].astype(int)
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=SEED, stratify=df["label_id"])
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["label_id"])
val_df = val_df.reset_index(drop=True); test_df = test_df.reset_index(drop=True)

# ── Load model + run inference ────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(ckpt)
model.eval()

@torch.no_grad()
def special_probs(frame):
    texts = frame["text"].astype(str).fillna("").tolist()
    probs = []
    for i in range(0, len(texts), 32):
        batch = tokenizer(texts[i:i+32], truncation=True, max_length=MAX_LEN,
                          padding=True, return_tensors="pt")
        logits = model(**batch).logits
        p = torch.softmax(logits, dim=-1)[:, SPECIAL]
        probs.extend(p.tolist())
    return np.array(probs)

val_p, val_y = special_probs(val_df), val_df["label_id"].values
test_p, test_y = special_probs(test_df), test_df["label_id"].values

def metrics_at(probs, y, t):
    pred = (probs >= t).astype(int)
    p, r, f, _ = precision_recall_fscore_support(
        y, pred, labels=[SPECIAL], average=None, zero_division=0)
    return {
        "threshold": round(float(t), 3),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
        "special_precision": round(float(p[0]), 4),
        "special_recall": round(float(r[0]), 4),
        "special_f1": round(float(f[0]), 4),
        "cm": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }

# ── Sweep thresholds on VALIDATION ────────────────────────────────────────
grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
val_sweep = [metrics_at(val_p, val_y, t) for t in grid]
print("\nVALIDATION sweep (threshold -> Special P/R/F1, macro-F1):")
for m in val_sweep:
    print(f"  t={m['threshold']:.2f}  P={m['special_precision']:.2f} "
          f"R={m['special_recall']:.2f} F1={m['special_f1']:.2f}  macroF1={m['macro_f1']:.2f}")

TARGET_RECALL = 0.70
# Among thresholds meeting target recall on val, take the largest (best precision).
meets = [m for m in val_sweep if m["special_recall"] >= TARGET_RECALL]
t_recall = max(meets, key=lambda m: m["threshold"])["threshold"] if meets else min(grid)
# Threshold that maximizes val macro-F1.
t_macro = max(val_sweep, key=lambda m: m["macro_f1"])["threshold"]
print(f"\nChosen on validation: t_recall>=0.70 -> {t_recall:.2f} | best-macro-F1 -> {t_macro:.2f}")

# ── Report on TEST at chosen thresholds + the 0.50 default ────────────────
report = {
    "checkpoint": ckpt,
    "target_recall": TARGET_RECALL,
    "thresholds_chosen_on_val": {"target_recall_0.70": t_recall, "max_macro_f1": t_macro},
    "test": {
        "default_0.50": metrics_at(test_p, test_y, 0.50),
        "target_recall_0.70": metrics_at(test_p, test_y, t_recall),
        "max_macro_f1": metrics_at(test_p, test_y, t_macro),
    },
}
print("\n=== TEST SET at each operating point ===")
for name, m in report["test"].items():
    print(f"  [{name}] t={m['threshold']:.2f}  acc={m['accuracy']:.3f} "
          f"macroF1={m['macro_f1']:.3f}  Special P={m['special_precision']:.2f} "
          f"R={m['special_recall']:.2f} F1={m['special_f1']:.2f}  cm={m['cm']}")

with open("evaluation_results_calibrated.json", "w") as f:
    json.dump(report, f, indent=2)

# Confusion matrix at the target-recall operating point
best = report["test"]["target_recall_0.70"]
disp = ConfusionMatrixDisplay(confusion_matrix=np.array(best["cm"]),
                              display_labels=["0", "1"])
fig, ax = plt.subplots(figsize=(7, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Class-weighted + calibrated (t={best['threshold']:.2f}) — Test Set")
plt.tight_layout()
plt.savefig("confusion_matrix_calibrated.png", dpi=150)
print("\nSaved: evaluation_results_calibrated.json, confusion_matrix_calibrated.png")
print("DONE")
