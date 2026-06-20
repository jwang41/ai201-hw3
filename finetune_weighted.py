"""
Re-fine-tune DistilBERT for TakeMeter with CLASS-WEIGHTED loss to fix the
Special-class imbalance (recall was 0.26 in the unweighted run).

Reproduces the exact same seeded, stratified 70/15/15 split as the notebook so
the test metrics are directly comparable to the 0.925 / recall-0.26 baseline.
CPU-only run.
"""
import torch  # MUST import before numpy/pandas on this box (else c10.dll WinError 1114)
import torch.nn as nn
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, f1_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

LABEL_MAP = {"0": 0, "1": 1}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)
MODEL_NAME = "distilbert-base-uncased"

# ── Data (identical to notebook) ──────────────────────────────────────────
df = pd.read_csv("./data/menu_item.csv")
df["label"] = df["label"].astype(str)
df["label_id"] = df["label"].map(LABEL_MAP)
df = df.dropna(subset=["label_id"])
df["label_id"] = df["label_id"].astype(int)

train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=SEED, stratify=df["label_id"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["label_id"]
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
print(f"Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}")
print("Train label counts:", train_df["label"].value_counts().to_dict())

# ── Class weights (inverse frequency) ─────────────────────────────────────
counts = train_df["label_id"].value_counts().sort_index().values
weights = counts.sum() / (NUM_LABELS * counts)
class_weights = torch.tensor(weights, dtype=torch.float)
print("Class weights:", class_weights.tolist())

# ── Tokenize (no `datasets` lib needed) ───────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

MAX_LEN = 96  # menu descriptions are short; keeps CPU training within time budget

class MenuDataset(torch.utils.data.Dataset):
    def __init__(self, frame):
        texts = frame["text"].astype(str).fillna("").tolist()
        self.enc = tokenizer(texts, truncation=True, max_length=MAX_LEN)
        self.labels = frame["label_id"].tolist()
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item

train_ds, val_ds, test_ds = MenuDataset(train_df), MenuDataset(val_df), MenuDataset(test_df)
collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ── Weighted Trainer ──────────────────────────────────────────────────────
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(outputs.logits.device))
        loss = loss_fct(outputs.logits.view(-1, NUM_LABELS), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS, id2label=ID_TO_LABEL, label2id=LABEL_MAP,
)

args = TrainingArguments(
    output_dir="./takemeter-model-weighted",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",   # select on macro-F1, not accuracy
    logging_steps=25,
    report_to="none",
    seed=SEED,
)

trainer = WeightedTrainer(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=val_ds,
    data_collator=collator, compute_metrics=compute_metrics,
)

print("Training (class-weighted, select on macro-F1)...")
trainer.train()

# ── Evaluate on full test set ─────────────────────────────────────────────
out = trainer.predict(test_ds)
pred_ids = np.argmax(out.predictions, axis=-1)
true_ids = out.label_ids
acc = accuracy_score(true_ids, pred_ids)
macro = f1_score(true_ids, pred_ids, average="macro", zero_division=0)
print(f"\n=== WEIGHTED MODEL — TEST SET ({len(true_ids)} items) ===")
print(f"Accuracy: {acc:.3f}   Macro-F1: {macro:.3f}")
label_names = [ID_TO_LABEL[i] for i in range(NUM_LABELS)]
report = classification_report(true_ids, pred_ids, target_names=label_names, zero_division=0)
print(report)
cm = confusion_matrix(true_ids, pred_ids)
print("Confusion matrix [rows=true, cols=pred]:\n", cm)

# Save artifacts
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
fig, ax = plt.subplots(figsize=(7, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Weighted Fine-Tuned Model — Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig("confusion_matrix_weighted.png", dpi=150)

rep_dict = classification_report(true_ids, pred_ids, target_names=label_names,
                                 zero_division=0, output_dict=True)
with open("evaluation_results_weighted.json", "w") as f:
    json.dump({
        "model": MODEL_NAME,
        "method": "class-weighted CE loss, 3 epochs, max_len 96, best-on-macro-F1",
        "test_set_size": int(len(true_ids)),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro), 4),
        "special_recall": round(float(rep_dict["1"]["recall"]), 4),
        "special_precision": round(float(rep_dict["1"]["precision"]), 4),
        "special_f1": round(float(rep_dict["1"]["f1-score"]), 4),
        "confusion_matrix": cm.tolist(),
    }, f, indent=2)
print("\nSaved: confusion_matrix_weighted.png, evaluation_results_weighted.json")
print("DONE")
