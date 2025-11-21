# Model Upgrade Proposal: Multi-Sweep Training + Tier-1 Architecture Enhancements

## 1. Overview
The current NEJM-style ultrasound gestational age model has several structural limitations that restrict predictive performance. This document outlines existing weaknesses, the proposed improvements, and the computational implications of using a more modern, sweep-aware, attention-driven architecture.

---


## 2. Limitations of the Current System

### 2.1 Single-Sweep Training
Only one random sweep is used during training, while inference uses all sweeps.  
This causes a significant distribution mismatch:

- The model never learns sweep-to-sweep variation.
- Probe angle, fetal pose, and motion differences are unseen during training.
- Leads to overly optimistic train loss and significantly worse validation/test performance.



### 2.2 Weak Temporal Attention
The current attention mechanism: 512 → 64 → 1

Limitations:
- Single-head attention captures only one temporal pattern.
- No separation between sweeps (all frames flattened together).
- Cannot model hierarchical temporal structure (frames → sweep → study).


### 2.3 Weak Backbone (ResNet18)
- Only 11M parameters.
- Limited feature depth for subtle anatomical structures.
- Underfits fine-grained ultrasound patterns (speckle, contour transitions, shadowing).

### 2.4 Minimal Regression Head
- A single fully-connected layer.
- Cannot model nonlinear biological variability.
- Weak gradient flow leading to slower, less stable training.

---

## 3. Proposed Improvements

### 3.1 Data Strategy: Two Random Sweeps per Study

**Training Step:**
- Instead of selecting 1 sweep per study, randomly select **2 sweeps**.
- For each sweep, sample *T* frames (random or uniformly spaced).

**Benefits:**
- Doubles variation seen during training.
- Reduces distribution mismatch between training and evaluation.
- Forces model to learn probe-angle variation.
- Improves generalization and robustness to noise.

**Cost:**
- ~2× more frames processed per batch.
- Easily manageable on an RTX 5090 (16GB).

### 3.2 Stronger Backbone (Tier-1 Upgrade)

#### Option A — ResNet50
- ~4× deeper than ResNet18.
- Drop-in replacement.
- Improved anatomical feature extraction.

#### Option B — ConvNeXt-Tiny (Recommended)
- Modern architecture with better spatial inductive bias.
- Significantly stronger performance on medical imaging.
- ~28M parameters with excellent efficiency.

**Cost:**
- ~1.8–2.2× more GPU memory.
- Fit comfortably on a 5090.

### 3.3 Sweep-Level Attention (Hierarchical Temporal Modeling)

Current issue: sweeps are flattened, destroying sweep identity.

**Proposed upgrade:**
1. Extract frame features per sweep.
2. Apply **intra-sweep attention** to summarize each sweep.
3. Apply **inter-sweep attention** across sweep embeddings.

Frames → Intra-Sweep Attention → Sweep Embeddings → Inter-Sweep Attention

**Benefits:**
- Maintains sweep identity.
- Learns meaningful grouping of frames within sweeps.
- Models anatomical variation between sweeps as an explicit relationship.

**Cost:**
- Minimal overhead (~5–10%).


### 3.4 Multi-Head Temporal Attention

Replace single-head attention with:

MultiHeadAttention(num_heads = 4 or 8)

**Benefits:**
- Learns multiple temporal patterns:
  - Fetal shape cues  
  - Sweep motion smoothness  
  - Noise variation  
  - Anatomical transitions  
- Provides better stability and improved feature richness.

**Cost:**
- ~20–30% additional computation.

### 3.5 Stronger Regression Head (MLP)

Replace single FC with:
512 → 256 → 128 → 1
Activation: GELU
Regularization: dropout


**Benefits:**
- Better nonlinearity modeling.
- Stronger gradients.
- More stable training dynamics.

**Cost:**
- Virtually negligible (<1M params).

---

## 4. Expected Improvements

| Component | Expected Gain |
|----------|----------------|
| 2-random-sweep training | +10–25% generalization |
| ConvNeXt-Tiny backbone | +10–15% feature quality |
| Sweep-level attention | +5–10% temporal reasoning |
| Multi-head attention | +5–10% robustness |
| MLP regression head | +3–5% stability |

**Total potential improvement: 20–40% reduction in MAE/MSE.**

Combined gains produce a model that:
- generalizes better,
- handles sweep variation robustly,
- learns deeper temporal representation,
- extracts stronger spatial features,
- converges faster and more stably.

---

## 5. Computational Summary

| Upgrade | GPU Cost | Worth It? |
|---------|----------|-----------|
| 2 sweeps | ~2× | ✔ Absolutely |
| ConvNeXt-Tiny | ~2× | ✔ Major gain |
| Sweep-level attention | +10% | ✔ Important |
| Multi-head attention | +25% | ✔ High impact |
| MLP | negligible | ✔ Free accuracy |

The proposed pipeline fits easily within **RTX 5090 16GB** that has been provided.

---

## 6. Conclusion
The enhanced architecture significantly improves sweep diversity exposure, temporal modeling, spatial feature extraction, and regression stability. These changes address fundamental flaws in the existing system and transition the model into a modern, sweep-aware, attention-enhanced gestational age estimation framework.

The combination of:
- two-sweep training,
- ConvNeXt backbone,
- hierarchical attention,
- multi-head temporal reasoning,
- and a stronger regression head unlocks a substantial leap in overall model accuracy, robustness, and real-world performance.

