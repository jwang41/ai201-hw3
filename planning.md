# Planning: TakeMeter — Menu-Item "Special vs Standard" Classifier

*Working notes. These are the scope decisions made before building; the final
report is in `README.md`.*

## 1. Project Overview

This document plans **TakeMeter**, a text classifier that reads a restaurant
menu-item description and decides whether the item is a **standard everyday
offering** or a **special / signature / unusual offering**. The intended
downstream use is a **menu-aggregation service**: a tool that automatically
surfaces a restaurant's distinctive and limited-time items instead of letting
them get buried among staples like fries and iced tea.

This document fixes the scope decisions (domain, label set, edge cases, data,
metrics, success criteria, AI-tool usage) before any annotation or modeling
begins.

---

## 2. Domain / Community

### 2.1 Choice

I chose **restaurant menu item descriptions** aggregated across many
restaurants (the `menu_item.csv` dataset: item name, free-text description,
price, and a binary label). The "community" here is the body of menus a food
aggregator ingests.

### 2.2 Why this domain is a good fit for classification

- **There is a real, concrete use for the labels.** A menu aggregator wants to
  highlight what makes each restaurant worth visiting — its specials and
  signature dishes — not its generic items. Auto-tagging items as
  *Special* vs *Standard* maps directly onto a product feature rather than being
  invented for the exercise.
- **The discourse is genuinely varied, not monotone.** Menu descriptions range
  from terse ("Iced Tea") to elaborate marketing copy ("Smoked Double Bacon
  Breakfast SOURDOUGH KING…"), across cuisines, meal types, and writing styles.
  That variety in length, vocabulary, and tone is what makes the classifier
  non-trivial.
- **The label is about intent, not keywords.** A "standard" side and a
  "special" creation often share the same ingredient words (cheese, sauce,
  grilled, *served with…*). The classifier cannot key off a word list; it has to
  learn the subtler signal of what makes a dish distinctive. That gap between
  surface features and the true label is what makes the task a fair test of a
  model.
- **Plentiful and well-bounded.** The dataset is sizable (~2,500 items), in
  English, and the label space is closed (two classes), so the task stays
  learnable instead of sprawling.

---

## 3. Labels

I use **two** mutually-exclusive labels, assigned by the *primary character* of
the item.

### 3.1 `0` — Standard
A standard menu offering: a common main course, side, drink, or everyday item
that most comparable restaurants would also carry.

- *Example A:* "Iced Tea — brewed fresh daily. Also available unsweetened."
- *Example B:* "Green Salad — lettuce, tomatoes, onions, pickles, and shredded
  cheese."

### 3.2 `1` — Special
A special, limited-time, signature, or otherwise unusual dish that distinguishes
this restaurant from a typical menu.

- *Example A:* "Breakfast Salsa."
- *Example B:* "Herb rawtilla topped with marinara, almond cheese, pesto and
  bruschetta tomatoes."

---

## 4. Hard Edge Cases

### 4.1 The core ambiguity: `Standard` vs `Special`
The boundary is inherently subjective. The hardest posts are **elaborately
described but ordinary dishes** (a richly-worded burger is still a burger) and
**plainly described but unusual dishes** (a one-line name for a regional
specialty). Surface richness of the text is a misleading cue in both directions.

**Decision rule (tie-breaker):** classify by *the dish itself, not the prose*.
Ask: "Would a typical restaurant of this type also carry this item?" If yes →
`Standard`, however fancy the wording. If the item is a signature, regional,
fusion, limited-time, or otherwise atypical dish → `Special`, however terse the
wording. When genuinely 50/50, default to `Standard` (the majority, lower-risk
class for the highlight feature).

### 4.2 Other anticipated edge cases
- **Empty / missing descriptions.** Many rows have a `name` but a blank `text`
  field. Rule: such rows carry almost no signal for a text model. They are kept
  in the dataset honestly (they exist in production) but flagged, and their
  impact is examined in error analysis rather than hidden.
- **Combo / "choice of" items.** "Choice of ground beef or shredded chicken,
  steak or grilled chicken." Rule: judge by the overall offering, not by the
  presence of multiple options.

### 4.3 Handling during annotation
- Maintain a living **edge-case log**: every item that takes more than ~15
  seconds to decide is recorded with the chosen label and reason.
- Apply the tie-breaker consistently; when a new pattern appears that the rules
  don't cover, pause, add a rule, and re-check earlier borderline items.
- A small **double-annotated subset** measures whether the rules actually
  produce agreement (Cohen's κ), which also sets the human performance ceiling.

---

## 5. Data Collection Plan

### 5.1 Source
The dataset is `data/menu_item.csv` — aggregated restaurant menu items with
columns `name`, `text`, `price`, `label`. The classifier reads the `text`
(description) field. Rows are used as-is, including empty descriptions, so the
evaluation reflects real production data.

### 5.2 Volume & splits
- **~2,491 labeled items total**, split **70 / 15 / 15** into train / validation
  / test (≈1,743 / 374 / 374).
- Splits are **stratified on the label** with a fixed seed so the class ratio is
  preserved across all three splits and the test set stays locked for honest
  evaluation.

### 5.3 Handling class imbalance
The dataset is **heavily imbalanced — only ~9% of items are `Special`** (225 of
2,491). This is the central challenge of the project. Plan:

1. **Keep the natural distribution in the test set** so metrics reflect reality;
   never balance the test set artificially.
2. **Lead with imbalance-aware metrics** (§6) so a majority-class-biased model
   cannot hide behind accuracy.
3. **If the `Special` class is too sparse to learn**, apply training-time
   remedies — class-weighted loss, oversampling the minority class, or
   decision-threshold tuning on the `Special` probability — and document each.
   These are applied to *training only*, never to validation/test.

### 5.4 Quality control
A subset is **double-annotated** to compute inter-annotator agreement (κ). Low
agreement is a signal that the Standard/Special definition or the §4 tie-breaker
needs tightening before scaling up — especially likely here given how subjective
the boundary is.

---

## 6. Evaluation Metrics

Accuracy alone is **actively misleading** here: with 91% of items in the
`Standard` class, a model that almost always predicts `Standard` scores ~91%
accuracy while being useless for the one job that matters (finding specials). The
metric suite is chosen to expose exactly that failure.

| Metric | Why it's needed for this task |
|---|---|
| **Macro-averaged F1** | *Primary metric.* Weights both classes equally, so the rare `Special` class counts as much as `Standard`. Directly penalizes a model that ignores specials. |
| **`Special` recall (per-class recall)** | The product goal is to *catch the specials*. Missing a special (false negative) is the costly error; recall on class `1` is therefore the single most important diagnostic. |
| **Per-class precision** | Too many false-positive "specials" would drown the real ones in the highlight feed, so `Special` precision matters too. |
| **Confusion matrix** | Shows the exact Standard↔Special error split and whether the model is collapsing toward the majority class. |
| **Overall accuracy** | Reported for context only — explicitly **not** the optimization target. |
| **Inter-annotator agreement (κ)** | Sets the human ceiling; given a subjective boundary, it defines what "good" even means. |

Macro-F1 (and `Special` recall) is the headline; accuracy is reported but not
optimized.

A **zero-shot LLM baseline** (Groq `llama-3.3-70b-versatile`, prompt-only) is
run on the same locked test set so the fine-tuned DistilBERT model can be judged
against a no-training reference rather than in a vacuum.

**Baseline execution risk:** the Groq free tier has a daily token cap (100k
TPD), and a naive loop over all 374 test items will hit HTTP 429 partway through
— in which case the failed items must *not* be silently dropped, because doing so
leaves the baseline scored on a tiny, non-random prefix of the test set and makes
any comparison meaningless. Mitigations planned: add retry/back-off on 429,
reduce per-call tokens, run in smaller batches across time, or evaluate the
baseline on a *fixed random subset* sized to fit the quota and compare the
fine-tuned model on that same subset.

---

## 7. Definition of Success

### 7.1 What "genuinely useful" means
TakeMeter is a *suggestion* layer for a menu aggregator's highlight feature, with
a human able to confirm or override. It is useful when it catches most specials
without flooding the feed with false ones.

### 7.2 Concrete bars
- **The metric that matters:** `Special` **recall**, since the feature's whole
  point is to find specials.
- **Good enough to deploy as an auto-tagger:** `Special` recall **≥ 0.70** with
  `Special` precision high enough to keep the highlight feed clean — i.e.
  **macro-F1 ≈ 0.75+**.
- **Assistive-only (human confirms):** lower recall is tolerable *only if* the
  reviewer is told how many specials the model misses.
- **Human ceiling:** establish κ first; the model is not expected to exceed human
  consistency on a boundary this subjective.

### 7.3 What we will *not* accept
A high overall **accuracy** driven by the majority `Standard` class while
`Special` recall sits near the floor is an explicit **failure**, regardless of
the headline number. The deployment decision is made on macro-F1 and `Special`
recall, not on accuracy.

---

## 8. AI Tool Plan

This is an annotation-and-evaluation project, not an implementation project, so
there is no application code for an AI tool to generate. AI tools are used at the
three points where they genuinely strengthen the *labeling and analysis*
workflow. Every AI-assisted step is logged for the AI-usage disclosure.

### 8.1 Label stress-testing (before annotation)
- **Procedure:** Give the AI the two label definitions (§3) and the edge-case
  description (§4), then ask it to generate **5–10 menu descriptions that
  deliberately sit on the Standard/Special boundary** (e.g. an elaborately worded
  ordinary burger; a tersely named regional specialty).
- **What I do with the output:** I label each generated item using only my
  written rules. Any item I *cannot* classify cleanly is evidence the definition
  or tie-breaker is underspecified.
- **Action on failure:** Tighten §3/§4 **now**, then re-generate to confirm the
  ambiguity is resolved. Generated items are used only for definition-tuning and
  are **not** added to the real train/val/test data.

### 8.2 Annotation assistance (during labeling)
**Decision: use an LLM to pre-label batches, then human-review every item.**
- **Tool:** Claude (Claude 4.x family) given the label definitions and
  tie-breaker as a system prompt; it outputs a suggested label plus a one-line
  rationale.
- **Workflow:** The LLM pre-labels a batch; I review **100%** and correct it. The
  pre-label is a draft, never final — the human decision is authoritative.
- **Tracking for disclosure:** provenance columns record `pre_labeled_by`,
  `llm_suggested_label`, `final_label`, and `changed`, so I can report the count
  of AI-pre-labeled items and the human override rate (a very low override rate
  would signal rubber-stamping).
- **Guardrail:** the double-annotation QC subset and the held-out **test set**
  are labeled by a human *first*, blind to any LLM suggestion, so AI assistance
  cannot contaminate the κ measurement or the final evaluation.

### 8.3 Failure analysis (after evaluation)
- **Procedure:** Export the misclassified test items (text, true label,
  predicted label, model confidence) and ask the AI to **cluster the errors into
  recurring patterns** — e.g. majority-class bias, empty descriptions,
  ambiguous-boundary cases.
- **What I look for:** systematic confusions tied to the §4 edge cases,
  data-quality artifacts (empty `text`), and whether one class drives most of the
  error mass.
- **Verification (mandatory):** every proposed pattern is **checked by hand**
  against the actual items it claims to cover, and I count how many errors it
  explains. AI-proposed patterns are treated as *hypotheses*; only human-verified
  patterns reach the write-up, with supporting counts.

### 8.4 Disclosure
The final report's AI-usage section states which tool was used at each stage, the
count and override rate of AI-pre-labeled items, and that all reported failure
patterns were human-verified. No AI output is used as ground-truth data or as a
final label without human review.
