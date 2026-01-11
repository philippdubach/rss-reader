# HN Success Predictor: From Over-Engineering to Robust Simplicity

A technical retrospective on building a machine learning system to predict Hacker News virality, and the engineering lessons learned from pursuing simplicity over complexity.

---

## Abstract

The HN Success Predictor is a machine learning component embedded within a personal RSS reader application. Its purpose is to estimate the probability that a given article title would achieve "success" on Hacker News (defined as >100 points), enabling intelligent prioritization of content in the user's feed.

Over seven iterations, the system evolved from a baseline DistilBERT classifier (ROC AUC 0.654) through increasingly complex ensemble architectures—including a SBERT + RoBERTa hybrid—before ultimately converging on a simplified, regularized RoBERTa-only model (ROC AUC 0.685). The final architecture achieves comparable discrimination to the ensemble while reducing overfitting by 61%, eliminating external dependencies, and cutting inference latency in half.

This document chronicles the evolution, the mistakes made (including a significant data leakage incident), and the engineering philosophy that guided the final design: **complexity is a liability unless it pays measurable rent**.

---

## 1. Problem Definition & Data

### 1.1 The Prediction Task

Given only the **title** of an article, predict the probability that it would achieve ≥100 points if submitted to Hacker News. This is a binary classification task with inherent noise—identical titles can achieve vastly different outcomes based on submission timing, submitter reputation, and stochastic network effects.

**Why 100 points?** This threshold represents approximately the top 33% of submissions that gain any traction. It separates "front page material" from the mass of submissions that receive minimal engagement.

### 1.2 Dataset Construction

Data was collected from the HN Algolia API across a 36-month window:

| Category | Points Range | Sample Size | Label |
|----------|-------------|-------------|-------|
| Hits | ≥100 | 30,000 | 1 |
| Medium | 20-99 | 30,000 | 0 |
| Low | 1-19 | 30,000 | 0 |

**Total: ~90,000 posts**

The stratified sampling ensures class balance (~33% positive) while capturing the full spectrum of HN submissions. Posts with missing timestamps were excluded.

### 1.3 The Temporal Split (Critical Design Decision)

Early experiments used random train/test splits, which introduced **temporal data leakage**. HN has strong temporal correlations—certain topics trend together, writing styles evolve, and news cycles create clusters of similar content. A random split allows the model to "see the future" by training on posts temporally adjacent to test posts.

**V6+ Solution: Strict Temporal Split**

```
Train (70%): 2022-01-12 → 2024-03-28   [63,000 posts]
Val (15%):   2024-03-28 → 2024-09-24   [13,500 posts]  
Test (15%):  2024-09-24 → 2025-01-09   [13,500 posts]
```

This mirrors production reality: the model is always predicting on content from the future relative to its training data.

---

## 2. Architecture Evolution

### 2.1 V1: DistilBERT Baseline

**Hypothesis:** A lightweight transformer fine-tuned on title classification should establish a reasonable baseline.

**Architecture:**
- DistilBERT-base (66M parameters)
- Linear classification head
- Standard cross-entropy loss

**Results:**
- ROC AUC: **0.654**
- Training: ~20 minutes on T4

**Post-mortem:** Confirmed that transformers can extract meaningful signal from titles alone. The 0.654 AUC significantly exceeds random (0.5), validating the approach. However, DistilBERT's aggressive distillation may sacrifice nuance needed for this task.

---

### 2.2 V3.2: RoBERTa + Label Smoothing

**Hypothesis:** RoBERTa's more robust pre-training (no next-sentence prediction, more data, longer training) should capture subtler linguistic patterns.

**Architecture:**
- RoBERTa-base (125M parameters)
- Label smoothing (ε=0.1) to prevent overconfident predictions
- Class weighting for imbalanced data

**Results:**
- ROC AUC: **0.692** (+3.8 pp over V1)
- Training: ~30 minutes on T4

**Post-mortem:** Significant improvement. Label smoothing helped calibration. The model learned meaningful patterns: "Show HN" and technical deep-dives score higher; generic news aggregation scores lower.

---

### 2.3 V3.3: SBERT + RoBERTa Ensemble (The False Peak)

**Hypothesis:** Sentence embeddings (SBERT) capture semantic similarity that fine-tuned classification heads might miss. An ensemble combining semantic features with fine-tuned discrimination should outperform either alone.

**Architecture:**
- **Branch A:** SBERT (`all-mpnet-base-v2`) → 768-dim embedding → MLP classifier
- **Branch B:** RoBERTa-base → fine-tuned classification head
- **Fusion:** Weighted average of probabilities (α-tuned on validation)

**Results:**
- ROC AUC: **0.714** (+2.2 pp over V3.2)
- α (SBERT weight): 0.35

**The Problem (discovered later):** This evaluation used a **random split**, not temporal. The inflated AUC was an artifact of data leakage.

---

### 2.4 V6: The Reality Check (Temporal Split)

**Hypothesis:** Re-evaluate V3.3's ensemble architecture with proper temporal evaluation.

**Architecture:** Identical to V3.3

**Results with Temporal Split:**
- ROC AUC: **0.693** (-2.1 pp from V3.3's reported score)
- α (SBERT weight): **0.10** (not 0.35)
- Train AUC: 0.803
- **Overfitting Gap: 0.109**

**Critical Findings:**

1. **The V3.3 score was inflated by ~2.1 AUC points** due to temporal leakage
2. **SBERT contributed almost nothing** (α=0.10 means 90% of signal came from RoBERTa)
3. **Severe overfitting:** 11-point gap between train and test AUC

**Post-mortem:** This was a pivotal moment. The ensemble added:
- 100% latency overhead (two forward passes)
- 384MB additional model weight (SBERT)
- Dependency on sentence-transformers library
- Measurable deployment complexity

For what? 0.001 AUC points over the RoBERTa component alone. **SBERT was not paying rent.**

---

### 2.5 V7: Principled Simplification (Current)

**Hypothesis:** A well-regularized RoBERTa model can match the ensemble while being simpler, faster, and more robust to distribution shift.

**Architecture:**
- RoBERTa-base with **increased regularization**:
  - Dropout: 0.1 → **0.2** (hidden, attention, classifier)
  - Weight decay: 0.01 → **0.05**
  - Epochs: 4 → **3** (prevent memorization)
  - **Frozen layers:** Embeddings + layers 0-5 (transfer learning approach)
- Isotonic calibration on validation set
- **No SBERT**

**Trainable Parameters:**
```
Total:     124,647,170
Trainable:  52,442,114 (42.1%)
Frozen:     72,205,056 (57.9%)
```

**Results:**

| Metric | V6 (Ensemble) | V7 (RoBERTa-only) | Delta |
|--------|---------------|-------------------|-------|
| Test AUC | 0.6934 | 0.6847 | -0.87% |
| Train AUC | 0.8026 | 0.7271 | -9.4% |
| Overfit Gap | 0.109 | **0.042** | **-61%** |
| ECE | 0.056 | **0.043** | -23% |
| 95% CI | [0.684, 0.704] | [0.675, 0.695] | overlapping |

**V7 Wins:**
- ✓ **61% reduction in overfitting** — better generalization
- ✓ **Better calibration** (ECE 0.043 < 0.05 threshold)
- ✓ **Simpler deployment** — no SBERT dependency
- ✓ **Faster inference** — single model pass
- ✓ **Smaller footprint** — 500MB vs 900MB

**V7 Trade-off:**
- 0.87% lower test AUC (within confidence interval overlap — not statistically significant)

---

## 3. The SBERT Investigation: A Detailed Post-Mortem

### 3.1 Why Did SBERT Seem Valuable in V3.3?

The random split created two sources of leakage:

1. **Temporal clustering:** Posts about the same news event (e.g., "OpenAI releases GPT-4") appear in both train and test. SBERT's semantic embeddings perfectly match these near-duplicates.

2. **Stylistic evolution:** Writing styles on HN evolve. A random split mixes 2024 posts into 2022 training data, allowing the model to learn "future" stylistic patterns.

### 3.2 Why Does SBERT Fail with Temporal Split?

SBERT embeddings are **semantic similarity features**, not **predictive features**. Two posts can be semantically identical but achieve wildly different outcomes based on:
- Time of submission (weekend vs. Monday morning)
- Current news cycle (competing stories)
- Submitter's karma/history
- Random network effects

RoBERTa, fine-tuned end-to-end for classification, learns these subtle cues embedded in word choice, phrasing, and structure. SBERT, frozen and similarity-focused, cannot.

### 3.3 The Complexity Trap

The V3.3 → V6 transition illustrates a common ML pitfall:

> "We added complexity, saw better numbers, and assumed causation."

The true causal graph:
```
Random Split → Data Leakage → Inflated AUC
     ↓
"SBERT helps!"  ← WRONG CONCLUSION
```

The lesson: **Always validate improvements under production-realistic conditions.**

---

## 4. Calibration: Why Raw Logits Are Insufficient

### 4.1 The Problem

Neural networks trained with cross-entropy produce logits that, after softmax, are **not well-calibrated probabilities**. A prediction of 0.7 does not mean "70% of posts with this score are hits."

For a user-facing RSS reader, miscalibration is problematic:
- Users lose trust if "80% likely" posts consistently fail
- Threshold-based filtering becomes unreliable
- Probability-based ranking introduces systematic bias

### 4.2 Calibration Methods Evaluated

| Method | Val ECE | Test ECE | Notes |
|--------|---------|----------|-------|
| None (Raw) | 0.089 | 0.092 | Overconfident |
| Platt (Sigmoid) | 0.051 | 0.058 | Parametric, assumes shape |
| **Isotonic** | **0.031** | **0.043** | Non-parametric, best fit |

### 4.3 Isotonic Regression

Isotonic calibration learns a monotonic step function mapping raw probabilities to calibrated probabilities. It makes no parametric assumptions, allowing it to correct arbitrary distortions in the probability space.

```python
from sklearn.isotonic import IsotonicRegression

calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(val_probs, val_labels)
calibrated_test_probs = calibrator.predict(test_probs)
```

Post-calibration, the model's predicted probabilities closely match empirical frequencies—a predicted 0.4 means approximately 40% of such posts are hits.

---

## 5. Production Integration

### 5.1 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RSS Reader Application                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────────┐  │
│  │  Feed Parser  │────▶│  HN Predictor │────▶│   Display    │  │
│  │  (feed_parser)│     │  (hn_predictor)│     │  (dashboard) │  │
│  └──────────────┘     └──────────────┘     └─────────────┘  │
│                              │                               │
│                              ▼                               │
│                    ┌──────────────────┐                     │
│                    │  models/hn_model_v7/                   │
│                    │  ├── config.json                       │
│                    │  ├── isotonic_calibrator.joblib        │
│                    │  └── roberta/                          │
│                    └──────────────────┘                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Inference Engine (`hn_predictor.py`)

Key design decisions:

1. **Lazy Loading:** Model loads on first prediction, not import
2. **Device Detection:** Automatic GPU/MPS/CPU selection
3. **Batch Processing:** Efficient inference for feed-level scoring
4. **Singleton Pattern:** Single model instance across application

```python
class HNPredictor:
    DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "hn_model_v7"
    
    def predict_batch(self, titles: list[str]) -> list[float]:
        """Predict calibrated probabilities for a batch of titles."""
        # Tokenize
        inputs = self._tokenizer(titles, padding=True, truncation=True, ...)
        
        # Inference
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
        
        # Calibrate
        return self._calibrator.predict(probs.numpy())
```

### 5.3 User-Facing Output

The predictor provides confidence levels based on probability thresholds:

| Probability | Confidence | User Signal |
|-------------|------------|-------------|
| ≥0.70 | High | Strong candidate |
| ≥0.30 (threshold) | Medium | Worth reading |
| ≥0.20 | Low | Marginal |
| <0.20 | Very Low | Likely skip |

---

## 6. Lessons Learned

### 6.1 On Evaluation

> **"Your metric is only as good as your split."**

The V3.3 → V6 transition cost weeks of work investigating a phantom improvement. Future projects will:
- Default to temporal splits for time-series data
- Validate any improvement exceeds confidence interval overlap
- Be suspicious of gains that seem "too clean"

### 6.2 On Complexity

> **"Every component must pay rent."**

SBERT added:
- 100% latency increase
- 400MB model weight
- External dependency
- Deployment complexity

For: 0.001 AUC points (noise-level improvement)

**The tax was not worth the benefit.**

### 6.3 On Overfitting

> **"Train AUC is a vanity metric."**

V6's 0.803 train AUC looked impressive. Its 0.693 test AUC revealed the truth: the model had memorized training patterns that didn't generalize. V7's aggressive regularization sacrificed train performance for test robustness.

### 6.4 On Calibration

> **"Users don't care about AUC. They care about trust."**

A miscalibrated 0.8 prediction that fails repeatedly destroys user confidence. Isotonic calibration ensures predictions mean what they say.

---

## 7. Future Directions

### 7.1 Potential Improvements (Not Pursued)

- **Domain features:** URL domain, title length, punctuation patterns
- **Temporal features:** Hour of day, day of week (requires submission time)
- **Historical context:** Author history, topic recency

**Why not pursued:** Each adds complexity. The current system achieves acceptable performance with minimal moving parts. Improvements would need to justify their complexity cost.

### 7.2 Monitoring

The model will drift as HN's culture evolves. Recommended monitoring:
- Weekly calibration checks on recent predictions
- Quarterly retraining on fresh data
- Alert on ECE > 0.10 or AUC < 0.65

---

## Appendix A: Model Card

**Model:** HN Success Predictor V7

| Property | Value |
|----------|-------|
| Architecture | RoBERTa-base (regularized) |
| Parameters | 124.6M (42% trainable) |
| Training Data | 90K HN posts (2022-2025) |
| Test AUC | 0.6847 [0.675, 0.695] |
| ECE | 0.043 |
| Threshold | 0.302 |
| Inference | ~50ms/batch (CPU), ~5ms/batch (GPU) |
| Model Size | ~500MB |

---

## Appendix B: Reproduction

Training notebook: `archive/hn_transformer_colab-v7.ipynb`

```bash
# Colab setup
!pip install transformers datasets scikit-learn

# Run all cells sequentially
# Model artifacts saved to Google Drive
# Download hn_model_v7.zip for local deployment
```

---

## Appendix C: Version Changelog

| Version | Date | AUC | Key Change |
|---------|------|-----|------------|
| V1 | 2024-03 | 0.654 | DistilBERT baseline |
| V3.2 | 2024-06 | 0.692 | RoBERTa + label smoothing |
| V3.3 | 2024-08 | 0.714* | SBERT ensemble (*inflated) |
| V6 | 2024-12 | 0.693 | Temporal split (reality check) |
| V7 | 2025-01 | 0.685 | Regularized, simplified |

---

*Document generated: 2025-01-11*
*Author: Technical retrospective of the rss-reader HN prediction system*
