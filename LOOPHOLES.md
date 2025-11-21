# 🐛 Code Loopholes & Bugs

**Analysis Date:** November 19, 2025  
**Project:** GA Regression Challenge - Ultrasound Gestational Age Prediction

---

## 🔴 Critical Bugs (Must Fix Immediately)

### 1. **Validation Multi-Sweep Logic is Completely Broken**
**Severity:** 🔴 CRITICAL  
**Location:** `train.py`, lines 104-108  
**File:** `train_and_validate()` function

```python
sweeps = sweeps.to(device)
B, S, T, C, H, W = sweeps.shape  # B=batch, S=num_sweeps, T=frames
sweeps = sweeps.view(B, S * T, C, H, W)  # Flatten sweeps
labels = labels.float().to(device).unsqueeze(1)
outputs, _ = model(sweeps)
```

**Problem:**
- Treats 8 separate sweeps as one continuous 128-frame video (8 sweeps × 16 frames)
- Should make 8 predictions per sample and ensemble them
- Current approach completely breaks multi-sweep validation logic
- Attention mechanism gets wrong temporal structure

**Impact:** 
- Validation metrics are meaningless
- Model evaluation is incorrect
- Test inference will also be wrong

**Fix Required:**
```python
# Loop through each sweep, predict separately, then average
predictions = []
for s in range(S):
    sweep = sweeps[:, s, :, :, :, :]  # (B, T, C, H, W)
    pred, _ = model(sweep)
    predictions.append(pred)
predictions = torch.stack(predictions, dim=1)  # (B, S, 1)
outputs = predictions.mean(dim=1)  # (B, 1) - ensemble average
```

---

### 2. **Loss Calculation Bug in Training Metrics**
**Severity:** 🔴 CRITICAL  
**Location:** `train.py`, lines 74-77  
**File:** `train_and_validate()` function

```python
train_loss += loss.item() * frames.size(0)
train_mae_epoch += mae * frames.size(0)
train_mse_epoch += mse * frames.size(0)
```

**Problem:**
- `mae` and `mse` are already averaged over the batch (from `torch.mean`)
- Multiplying averaged values by batch size then dividing by total samples = incorrect weighting
- Loss accumulation is inconsistent

**Impact:**
- Reported training metrics are mathematically wrong
- Can't compare losses across different batch sizes
- TensorBoard logs are misleading

**Fix Required:**
```python
# Option 1: Accumulate raw losses
train_loss += loss.item() * frames.size(0)
train_mae_epoch += torch.sum(torch.abs(outputs - labels)).item()
train_mse_epoch += torch.sum((outputs - labels) ** 2).item()

# Option 2: Just average batch losses (simpler)
train_loss += loss.item()
train_mae_epoch += mae
train_mse_epoch += mse
# Then divide by number of batches, not total samples
```

---

### 3. **Model Checkpoint Format Causes Loading Issues**
**Severity:** 🔴 CRITICAL  
**Location:** `train.py`, lines 127-132  
**File:** `train_and_validate()` function

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'model_architecture': model,  # ⚠️ Saves entire model object
    'epoch': epoch + 1,
    'val_loss': val_loss
}, save_path)
```

**Problem:**
- Saving entire model object (`model_architecture: model`) causes:
  - Version incompatibility (can't load on different PyTorch versions)
  - Massive file size (includes optimizer state, gradients)
  - Code dependency issues (requires exact same model definition)
- `infer.py` line 19 expects this format: `model = checkpoint['model_architecture']`

**Impact:**
- Model can't be shared across environments
- Checkpoint files unnecessarily large
- Deployment becomes difficult

**Fix Required:**
```python
# Save only state_dict and config
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': {
        'reduced_dim': 128,
        'fine_tune_backbone': True,
        'pretrained': True
    },
    'epoch': epoch + 1,
    'val_loss': val_loss
}, save_path)

# In infer.py, load like this:
checkpoint = torch.load(model_path)
model = NEJMbaseline(**checkpoint['model_config'])
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## 🟠 Major Issues (Should Fix Soon)

### 4. **Data Loading Race Condition - No Random Seed**
**Severity:** 🟠 MAJOR  
**Location:** `data.py`, line 40  
**File:** `SweepDataset.__getitem__()` method

```python
path = random.choice(row[self.sweep_cols])
```

**Problem:**
- Uses Python's `random.choice()` without seeding
- Different DataLoader workers get different random states
- Results are non-reproducible across runs
- Can't debug or compare experiments reliably

**Impact:**
- Non-deterministic training
- Can't reproduce results
- Makes debugging nearly impossible

**Fix Required:**
```python
# In DataLoader initialization, add worker_init_fn
def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % 2**32)
    random.seed(torch.initial_seed() % 2**32)

train_loader = DataLoader(..., worker_init_fn=worker_init_fn)

# Or in dataset, use numpy random with proper seeding
def __getitem__(self, idx):
    np.random.seed(idx)
    path = np.random.choice(row[self.sweep_cols])
```

---

### 5. **Index Alignment Risk in Inference**
**Severity:** 🟠 MAJOR  
**Location:** `infer.py`, lines 42-44  
**File:** `infer_test()` function

```python
start_idx = i * test_loader.batch_size
end_idx = min(start_idx + B, len(test_df))
study_ids.extend(test_df.iloc[start_idx:end_idx]['study_id'].tolist())
```

**Problem:**
- Assumes batch iteration order matches DataFrame order
- If `shuffle=True` or `drop_last=True` was set, indices won't align
- Predictions get matched to wrong study IDs
- Silent data corruption

**Impact:**
- Wrong predictions saved to wrong study IDs
- Evaluation metrics become meaningless
- Could affect clinical decisions in real deployment

**Fix Required:**
```python
# Option 1: Return study_id from dataset
class SweepEvalDataset:
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_id = row['study_id']
        # ... process data ...
        return all_sweeps, label, study_id

# Option 2: Use batch indices properly
for batch_idx, (sweeps, labels, study_ids_batch) in enumerate(test_loader):
    # study_ids_batch comes from dataset
```

---

### 6. **Missing NaN/Inf Validation in Medical Images**
**Severity:** 🟠 MAJOR  
**Location:** `data.py`, lines 44-64  
**Files:** `SweepDataset.__getitem__()` and `SweepEvalDataset.__getitem__()`

```python
img = nib.load(path).get_fdata().astype(np.float32)
```

**Problem:**
- No validation that NIfTI files loaded correctly
- Medical images can have NaN, Inf, or extreme outliers
- No intensity clipping or normalization before transforms
- Negative values break ImageNet normalization assumptions

**Impact:**
- Training crashes silently or gets NaN gradients
- Model learns from corrupted data
- Predictions become unreliable

**Fix Required:**
```python
img = nib.load(path).get_fdata().astype(np.float32)

# Validate and clean data
if np.any(np.isnan(img)) or np.any(np.isinf(img)):
    raise ValueError(f"Corrupted NIfTI file: {path}")

# Clip and normalize intensity
img = np.clip(img, np.percentile(img, 1), np.percentile(img, 99))
img = (img - img.min()) / (img.max() - img.min() + 1e-8)
```

---

### 7. **Attention Weights Computed But Never Used**
**Severity:** 🟠 MAJOR  
**Location:** Throughout codebase  
**Files:** `train.py`, `infer.py`, `model.py`

```python
outputs, attn_weights = model(frames)  # attn_weights ignored
```

**Problem:**
- Attention mechanism computes weights but they're never:
  - Logged to TensorBoard for visualization
  - Used for interpretability/debugging
  - Regularized (e.g., entropy loss for diversity)
  - Saved for clinical interpretation
- Wasted computation and missed insights

**Impact:**
- Can't understand what frames model focuses on
- No interpretability for medical validation
- Missing debugging information

**Fix Required:**
```python
# In training loop
outputs, attn_weights = model(frames)
writer.add_histogram('Attention/Weights', attn_weights, global_step)

# Add entropy regularization
entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=1).mean()
loss = criterion(outputs, labels) - 0.01 * entropy  # Encourage diverse attention

# Save attention maps for visualization
torch.save(attn_weights.cpu(), f'attention_epoch{epoch}.pt')
```

---

### 8. **No Exception Handling in Dataset Loading**
**Severity:** 🟠 MAJOR  
**Location:** `data.py`, `__getitem__()` methods  
**Files:** Both `SweepDataset` and `SweepEvalDataset`

```python
def __getitem__(self, idx):
    row = self.df.iloc[idx]
    path = random.choice(row[self.sweep_cols])
    img = nib.load(path).get_fdata()  # Can crash!
```

**Problem:**
- If NIfTI file is missing, corrupted, or unreadable → entire training crashes
- No try/except blocks
- No graceful degradation
- Can't skip bad samples

**Impact:**
- Training fails completely on one bad file
- Can't identify which samples are problematic
- Wastes hours of training time

**Fix Required:**
```python
def __getitem__(self, idx):
    try:
        row = self.df.iloc[idx]
        path = random.choice(row[self.sweep_cols])
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"NIfTI file not found: {path}")
        
        img = nib.load(path).get_fdata().astype(np.float32)
        # ... process ...
        return frames, label
    
    except Exception as e:
        print(f"Error loading sample {idx}: {e}")
        # Return next valid sample or skip
        return self.__getitem__((idx + 1) % len(self))
```

---

## 🟡 Medium Priority Issues

### 9. **Hard-Coded Frame Count (Magic Number)**
**Severity:** 🟡 MEDIUM  
**Location:** `data.py`, lines 50, 99  
**Files:** Both dataset classes

```python
target_frames = 16  # Magic number repeated
```

**Problem:**
- Frame count is hard-coded in multiple places
- Should be a configurable parameter
- Makes experimentation difficult
- Code duplication

**Fix Required:**
```python
class SweepDataset(Dataset):
    def __init__(self, csv_path, transform=None, target_frames=16):
        self.target_frames = target_frames
        # Use self.target_frames throughout
```

---

### 10. **Model State Not Protected During Validation**
**Severity:** 🟡 MEDIUM  
**Location:** `train.py`, line 86  
**File:** `train_and_validate()` function

```python
model.eval()
# ... validation code ...
# If exception occurs, model stays in eval mode
```

**Problem:**
- If validation crashes, model remains in eval mode
- Next training iteration has wrong behavior (no dropout, no batchnorm updates)
- Should use try/finally or context manager

**Fix Required:**
```python
try:
    model.eval()
    with torch.no_grad():
        # validation code
finally:
    model.train()  # Ensure model returns to train mode
```

---

### 11. **TensorBoard Step Counter Inconsistency**
**Severity:** 🟡 MEDIUM  
**Location:** `train.py`, line 110  
**File:** Validation logging

```python
writer.add_scalar("Val/Batch_Loss", loss.item(), global_step)
```

**Problem:**
- Validation uses training's `global_step` counter
- Should have separate validation step counter
- Causes weird x-axis alignment in TensorBoard
- Can't compare train/val curves properly

**Fix Required:**
```python
global_step = 0
val_step = 0

# In validation loop
writer.add_scalar("Val/Batch_Loss", loss.item(), val_step)
val_step += 1
```

---

### 12. **Incorrect Medical Trimester Calculation**
**Severity:** 🟡 MEDIUM  
**Location:** `evaluate2.py`, lines 33-34  
**File:** `compute_metrics_by_trimester()` function

```python
step = (max_ga - min_ga) / 3
trimester_bins = [min_ga, min_ga + step, min_ga + 2*step, max_ga]
```

**Problem:**
- Divides dataset range into 3 equal parts
- Medical trimesters are **fixed** periods, not data-dependent:
  - T1: 0-13 weeks (0-91 days)
  - T2: 14-26 weeks (92-182 days)  
  - T3: 27-40 weeks (183-280 days)
- Current method produces arbitrary, incorrect trimesters

**Impact:**
- Clinical interpretation is wrong
- Can't compare with medical literature
- Misleading performance analysis

**Fix Required:**
```python
# Medical trimester boundaries (in days)
trimester_bins = [0, 91, 182, 280]
trimester_labels = ['First Trimester (0-13 weeks)', 
                    'Second Trimester (14-26 weeks)', 
                    'Third Trimester (27-40 weeks)']
merged['trimester'] = pd.cut(y_true, bins=trimester_bins, 
                              labels=trimester_labels, include_lowest=True)
```

---

### 13. **No Input Validation on Batch Size**
**Severity:** 🟡 MEDIUM  
**Location:** `train.py`, function entry  
**File:** `train_and_validate()` function

**Problem:**
- No validation that:
  - CSV files exist before training starts
  - Batch size ≤ dataset size
  - Required columns are present
  - Paths in CSV are valid
- Fails late (after setup) instead of early

**Fix Required:**
```python
def train_and_validate(train_csv, val_csv, epochs=100, batch_size=8, ...):
    # Validate inputs
    assert os.path.exists(train_csv), f"Train CSV not found: {train_csv}"
    assert os.path.exists(val_csv), f"Val CSV not found: {val_csv}"
    
    train_df = pd.read_csv(train_csv)
    assert len(train_df) >= batch_size, f"Batch size {batch_size} > dataset size {len(train_df)}"
    assert 'ga' in train_df.columns, "Train CSV missing 'ga' column"
    # ... more validation
```

---

### 14. **Checkpoint Directory Creation Edge Case**
**Severity:** 🟡 MEDIUM  
**Location:** `train.py`, line 130  
**File:** Model saving

```python
os.makedirs(os.path.dirname(save_path), exist_ok=True)
```

**Problem:**
- If `save_path='model.pth'` (no directory), `dirname()` returns empty string
- Tries to create empty directory → undefined behavior
- Should check if dirname is non-empty

**Fix Required:**
```python
save_dir = os.path.dirname(save_path)
if save_dir:
    os.makedirs(save_dir, exist_ok=True)
```

---

### 15. **No Gradient Accumulation Support**
**Severity:** 🟡 MEDIUM  
**Location:** `train.py`, training loop  
**File:** `train_and_validate()` function

**Problem:**
- Video data is memory-intensive
- Might need effective batch size > GPU memory allows
- No support for gradient accumulation (common technique)
- Limits experimentation with larger batches

**Fix Required:**
```python
accumulation_steps = 4  # Effective batch = 8 * 4 = 32

for batch_idx, (frames, labels) in enumerate(train_loader):
    outputs, _ = model(frames)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()
    
    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 📊 Summary Statistics

| Severity | Count | Priority |
|----------|-------|----------|
| 🔴 Critical | 3 | **Fix Immediately** |
| 🟠 Major | 5 | Fix Soon |
| 🟡 Medium | 7 | Nice to Have |
| **Total** | **15** | - |

---

## 🎯 Recommended Fix Order

### Phase 1 (Critical - Do Now)
1. ✅ Fix validation multi-sweep logic (Bug #1)
2. ✅ Fix loss calculation (Bug #2)  
3. ✅ Fix checkpoint format (Bug #3)

### Phase 2 (Major - This Week)
4. Add NaN/Inf validation (Bug #6)
5. Fix random seed for reproducibility (Bug #4)
6. Add exception handling in dataset (Bug #8)
7. Fix study_id alignment in inference (Bug #5)

### Phase 3 (Medium - Next Sprint)
8. Use attention weights for interpretability (Bug #7)
9. Fix TensorBoard step counters (Bug #11)
10. Fix trimester calculations (Bug #12)
11. Add input validation (Bug #13)

### Phase 4 (Nice to Have)
12. Make target_frames configurable (Bug #9)
13. Add gradient accumulation (Bug #15)
14. Fix directory creation edge case (Bug #14)
15. Protect model state during validation (Bug #10)

---

## 🔧 Testing Checklist After Fixes

- [ ] Train for 2 epochs - verify no crashes
- [ ] Check validation metrics are reasonable
- [ ] Load saved checkpoint and run inference
- [ ] Verify study_id alignment in predictions
- [ ] Test with corrupted NIfTI file (should skip gracefully)
- [ ] Run with different random seeds - verify different results but similar convergence
- [ ] Check TensorBoard logs look correct
- [ ] Validate trimester metrics match medical definitions
- [ ] Test with batch_size=1 and batch_size=32
- [ ] Verify attention weights are logged

---

**Document Version:** 1.0  
**Last Updated:** November 19, 2025  
**Maintainer:** Development Team
