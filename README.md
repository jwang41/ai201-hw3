# TakeMeter — Menu-Item Classifier

**AI201 · Project 3 — Final Report**

TakeMeter is a text classifier that reads a restaurant menu-item description and
decides whether the item is a **standard offering** or a **special / unusual
offering**. The goal is a tool a menu-aggregation service could use to
automatically surface a restaurant's signature and limited-time items instead of
burying them among everyday staples.

This report summarizes what was built, how it was evaluated, and what the results
mean. It is self-contained — you do not need to read `planning.md` to follow it.

---

## 1. The Task

| | |
|---|---|
| **Domain** | Restaurant menu item descriptions (name + free-text description) |
| **Type** | Binary, single-label text classification |
| **Input** | The item's description text |
| **Output** | One of two labels: `0` (Standard) or `1` (Special) |
| **Use case** | Auto-tagging menu items so an aggregator can highlight specials |

### Why this is a real, non-trivial classification problem
The label is about the *character of the dish*, not its surface keywords. A plain
"served with rice" side and an elaborate chef's creation can share much of the
same vocabulary (cheese, sauce, grilled, served with…), so the model cannot
simply key off ingredient words — it has to learn the subtler signal of what
makes an item "special." That gap between surface features and the true label is
what makes the task worth a learned model rather than a keyword rule.

---

## 2. Labels

Each item is assigned **exactly one** of two labels:

- **`0` — Standard.** A standard menu offering: a common main course, side, or
  everyday item that most comparable restaurants would also carry.
  *Examples:* "Iced Tea — brewed fresh daily." · "Green Salad — lettuce,
  tomatoes, onions, pickles, and shredded cheese."

- **`1` — Special.** A special, limited-time, signature, or otherwise unusual
  dish that distinguishes this restaurant from a typical menu.
  *Examples:* "Breakfast Salsa." · "Herb rawtilla topped with marinara, almond
  cheese, pesto and bruschetta tomatoes."

---

## 3. Dataset

| Split | Examples | Label 0 (Standard) | Label 1 (Special) |
|---|---|---|---|
| **Total** | 2,491 | 2,266 (91%) | 225 (9%) |
| Train (70%) | 1,743 | 1,586 | 157 |
| Validation (15%) | 374 | — | — |
| **Test (15%)** | 374 | 340 | 34 |

- Source: `data/menu_item.csv` (columns: `name`, `text`, `price`, `label`).
- Splits are **stratified** on the label and produced with a fixed seed
  (`random_state=42`) so the class ratio is preserved across train/val/test and
  the test set stays locked for honest evaluation.
- **The dataset is heavily imbalanced** — only ~9% of items are `Special`. This
  single fact drives most of the results and analysis below.

---

## 4. Models

### 4.1 Fine-tuned model (primary)
- **Base:** `distilbert-base-uncased` with a 2-class classification head.
- **Training:** 3 epochs · learning rate `2e-5` · batch size 16 · weight decay
  `0.01` · 50 warmup steps · max sequence length 256 tokens.
- Best checkpoint selected on validation accuracy; defaults from the starter
  notebook were used unchanged.

### 4.2 Zero-shot baseline (Groq, `llama-3.3-70b-versatile`)
A prompt-only baseline that does no training: each test item is sent to the LLM
with a system prompt that defines both labels and gives one example apiece, and
the model is asked to return only the label name.

- **Model & decoding:** `llama-3.3-70b-versatile` · `temperature=0` (deterministic)
  · `max_tokens=20` · 0.1 s delay between calls.
- **Classification prompt** (system message):
  ```text
  You are classifying menu item descriptions from a restaurant's menu.
  Assign each post to exactly one of the following categories.

  0: This item is a standard menu offering, typically a main course or common side dish.
  Example: "Grilled salmon served with roasted vegetables and a lemon-dill sauce."

  1: This item is a special offering, a limited-time promotion, or a unique, unusual dish.
  Example: "Chef's special - pan-seared scallops with saffron risotto and asparagus."

  Respond with ONLY the label name.
  Do not explain your reasoning.

  Valid labels:
  0
  1
  ```
- **Output parsing:** the reply is lower-cased and matched against the label
  strings (longest first, exact-or-substring); a reply that matches neither `0`
  nor `1` is counted as unparseable and excluded from scoring.

> ⚠️ **The baseline run is not a valid comparison — it was crippled by API rate
> limits.** Of the 374 test items, only **42 received a prediction**; the other
> **332 failed with HTTP 429 rate-limit errors** (Groq free-tier cap of 100,000
> tokens/day) and were dropped. Because the loop runs in order, those 42 are just
> the *first* 42 rows of the test set — not a representative sample — and only 2
> of them were `Special`. The numbers below are reported for completeness but
> should **not** be read as a real head-to-head. See §5.2.

---

## 5. Evaluation Results (Test Set)

### 5.1 Fine-tuned DistilBERT — the real result

**Headline: 92.5% accuracy — but accuracy is the wrong story here.**

| Metric | Value |
|---|---|
| Overall accuracy | **0.925** (346 / 374 correct) |
| Macro-avg F1 | **0.68** |
| Macro-avg precision | 0.84 |
| Macro-avg recall | 0.63 |
| Weighted-avg F1 | 0.91 |

**Per-class breakdown**
| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `0` Standard | 0.93 | 0.99 | 0.96 | 340 |
| `1` Special | **0.75** | **0.26** | **0.39** | 34 |

**Confusion matrix** (see `confusion_matrix.png`)
|  | Pred `0` | Pred `1` |
|---|---|---|
| **True `0`** | 337 | 3 |
| **True `1`** | 25 | 9 |

**Reading the numbers.** The 92.5% accuracy is almost entirely a reflection of
the class imbalance: because 91% of items are `Standard`, a model that leans
toward predicting `Standard` looks excellent on accuracy while failing at the job
that matters. The class we actually care about — `Special` — has a recall of just
**0.26**, meaning the model **misses ~3 of every 4 special items** (25 of 34).
When it *does* call something special it is usually right (precision 0.75, only 3
false positives), but it is far too conservative to be useful. This precision-high
/ recall-low profile is exactly why macro-F1 (0.68) is the honest headline and
accuracy is not.

### 5.2 Baseline comparison (caveated)

| Model | Accuracy | Eval set | `Special` F1 |
|---|---|---|---|
| Fine-tuned DistilBERT | 0.925 | 374 / 374 | 0.39 |
| Groq zero-shot (Groq) | 0.929 | **42 / 374** | **0.00** (0 / 2) |

The notebook prints a "fine-tuning regression: 0.003," but that comparison is an
artifact: the two accuracies are computed on different samples (374 vs. 42 items)
and the baseline never even attempted 89% of the test set. On the handful of
`Special` items it *did* see, the baseline scored **0.00 F1** (got 0 of 2). The
only defensible reading is: **the baseline is inconclusive on accuracy and, on
the limited evidence available, no better than the fine-tuned model at the class
that matters.** To make the comparison valid, the baseline must be re-run within
rate limits (see §7).

### 5.3 Follow-up experiment — class-weighted re-fine-tune

The §5.1 model is too conservative: it misses ~3 of every 4 specials. Since the
diagnosis was class imbalance, I re-fine-tuned DistilBERT with a **class-weighted
cross-entropy loss** (weights ∝ inverse class frequency: Standard ×0.55, Special
×5.55), selecting the best checkpoint on **macro-F1** rather than accuracy. Same
seeded split, same locked 374-item test set; reproduced by
[`finetune_weighted.py`](finetune_weighted.py) (outputs
`evaluation_results_weighted.json` and `confusion_matrix_weighted.png`).

| Model | Accuracy | Macro-F1 | `Special` P | `Special` R | `Special` F1 |
|---|---|---|---|---|---|
| §5.1 unweighted | **0.925** | **0.68** | 0.75 | 0.26 | **0.39** |
| §5.3 class-weighted | 0.802 | 0.62 | 0.26 | **0.62** | 0.36 |

**Confusion matrix, weighted model** (`confusion_matrix_weighted.png`)
|  | Pred `0` | Pred `1` |
|---|---|---|
| **True `0`** | 279 | 61 |
| **True `1`** | 13 | 21 |

**What the re-fine-tune showed.** Weighting did exactly what it is designed to do:
`Special` **recall more than doubled, 0.26 → 0.62** (the model now catches 21 of
34 specials instead of 9). But it flipped the failure mode — the model now
*over*-predicts `Special`, so precision collapsed (0.75 → 0.26, with 61 false
positives) and both accuracy and macro-F1 fell. Net `Special` F1 is essentially
unchanged (0.39 → 0.36).

**Conclusion: neither model is deployable, and the trade-off is the finding.** One
model is silent about specials; the other is trigger-happy. Moving the loss weight
only slides the operating point along the precision/recall curve — it does not
create new signal. The real bottleneck is the **tiny, partly-noisy `Special`
class (157 training examples)**, so the highest-value next steps are *data* (more
and cleaner `Special` examples) and *threshold calibration*, not just reweighting
(see §7).

---

## 6. Error Analysis

I reviewed the 28 misclassified test items (the notebook prints them with model
confidence). Three patterns explain the errors; each was verified by reading the
underlying examples.

**Pattern 1 — Majority-class bias (the dominant failure).**
25 of 28 errors are `Special` items predicted as `Standard`, many with very high
confidence (0.89–0.98). Examples: a Polish/jalapeño kielbasa on a brioche bun
(conf 0.93), Oaxaca-cheese tacos with tomatillo sauce (0.92), an
egg-sausage-cheddar "blueberry maple square" (0.97). Having seen ~10× more
`Standard` items in training, the model defaults to `Standard` whenever a
description contains ordinary ingredient words — even when the dish is genuinely
unusual. This is the direct consequence of the 9% / 91% imbalance, and it is why
`Special` recall is only 0.26.

**Pattern 2 — Empty / missing descriptions.**
Two test errors have `text` literally equal to `"nan"` (the item has a name but
no description). These were labeled `Special` but predicted `Standard` with 0.98
confidence. With no text to read, the model cannot possibly classify them — this
is a **data-quality problem**, not a modeling one. Many CSV rows have an empty
`text` field, so the model is sometimes asked to classify nothing.

**Pattern 3 — Genuinely ambiguous boundary / label noise.**
The 3 false positives are `Standard` items predicted `Special` (e.g. "Served with
lettuce, tomatoes and cheese," conf 0.52; "Choice of ground beef or shredded
chicken, steak or grilled chicken," conf 0.60) — low-confidence calls on items
whose labeling is itself debatable. The Standard/Special line is subjective, and a
human annotator could reasonably disagree, so some "errors" are really
disagreements with noisy labels.

**Takeaway:** the model's weakness is not language understanding — it is the
imbalance (Pattern 1), compounded by empty inputs (Pattern 2) and a fuzzy label
definition (Pattern 3).

---

## 7. Is It Good Enough? (Definition of Success)

For the intended use — automatically surfacing a restaurant's special items — the
metric that matters is **recall on `Special`**: we want to catch the specials,
and a few false positives are tolerable.

- **Current state:** `Special` recall = 0.26. The tool would miss roughly
  three-quarters of specials, so it is **not good enough to deploy** as an
  automatic tagger, despite the flattering 92.5% accuracy.
- **A reasonable bar for usefulness** would be `Special` recall ≥ 0.70 while
  keeping precision high enough to avoid drowning real specials in false
  positives — i.e. a macro-F1 in the ~0.75+ range rather than today's 0.68.
- **As an assistive tool** (surfacing *candidate* specials for a human to
  confirm) the current model has limited value, and only if the reviewer knows it
  misses most specials.

### What I would do next
1. **Imbalance — tried, partly works (see §5.3).** Class-weighted loss lifted
   `Special` recall 0.26 → 0.62 but tanked precision to 0.26. Reweighting alone
   only moves the operating point; the next lever is **threshold calibration**
   (pick the `Special` probability cutoff that hits a target recall at acceptable
   precision) rather than more weighting.
2. **Get more / cleaner `Special` data** — with only 157 `Special` training
   examples, this is the real ceiling. More labeled specials would help both
   recall and precision in a way reweighting cannot.
3. **Clean the data** — drop or repair rows with empty `text`; consider using the
   `name` field as an extra signal.
4. **Tighten the label definition** and re-check the noisiest examples, since some
   "errors" are really annotation disagreements.
5. **Re-run the baseline within rate limits** — add retry/back-off on HTTP 429,
   batch fewer tokens, or evaluate on a fixed random subset — so there is a valid
   zero-shot reference to compare against.

---

## 8. Reproducing the Results

1. Open `takemeter_starter.ipynb` in Google Colab with a **T4 GPU** runtime.
2. Make `menu_item.csv` available at the path used in Section 1.
3. Run Sections 1–4 to load the data, fine-tune DistilBERT, and produce the test
   metrics and `confusion_matrix.png`.
4. For the baseline (Sections 5–6), set a `GROQ_API_KEY` (Colab Secrets) and run
   the cells. **Mind the free-tier daily token cap** — without back-off the run
   will hit HTTP 429 partway through (as it did here). Metrics are written to
   `evaluation_results.json`.

The §5.3 class-weighted re-fine-tune runs **locally on CPU** (no Colab needed):
```bash
pip install -r requirements.txt
python finetune_weighted.py        # writes evaluation_results_weighted.json + confusion_matrix_weighted.png
```
> Note: on Windows, `import torch` must come before `numpy`/`pandas` or torch's
> `c10.dll` fails to load (WinError 1114); the script already orders imports this
> way.

### Repository layout
```
ai201-hw3/
├── README.md                          # this report
├── planning.md                        # working notes / design doc
├── takemeter_starter.ipynb            # training + evaluation pipeline (Colab)
├── finetune_weighted.py               # §5.3 class-weighted re-fine-tune (local CPU)
├── config.py                          # Groq model + data path settings
├── data/menu_item.csv                 # labeled dataset (2,491 items)
├── confusion_matrix.png               # unweighted model (§5.1)
├── confusion_matrix_weighted.png      # class-weighted model (§5.3)
├── evaluation_results.json            # unweighted metrics
└── evaluation_results_weighted.json   # class-weighted metrics
```

---

## 9. AI Tool Usage

Consistent with the AI-tool plan in `planning.md`, AI assistance was used for:
- **Label stress-testing** — probing the Standard/Special boundary with generated
  edge cases before committing to the definitions.
- **Error-pattern analysis** — clustering the 28 misclassifications into the three
  patterns in §6, each of which was then verified by hand against the actual
  examples (no pattern was reported without reading the underlying items).

No AI output was used as ground-truth labels or accepted as a finding without
human verification.
