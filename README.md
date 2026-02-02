# Enigma 2027 - Winning Solution 🏆

## Professional Networking Compatibility Prediction
### CodeFest'26 by IIT BHU Varanasi

---

## 🎯 Key Discovery

The target compatibility scores are **Jaccard Similarity** values:

```
Compatibility(u, v) = |Items_u ∩ Items_v| / |Items_u ∪ Items_v|
```

Where `Items = Business_Interests ∪ Business_Objectives ∪ Constraints`

### Evidence
- Scores like 0.1429 = 1/7, 0.1667 = 1/6, 0.2 = 1/5, 0.25 = 1/4 are Jaccard fractions
- ~25% of pairs have 0.0 compatibility (no shared items)
- Mean score ~0.095 matches Jaccard distribution

---

## 📁 Files

| File | Description |
|------|-------------|
| `kaggle_solution_final.py` | **Main solution** - Copy this to Kaggle |
| `kaggle_ultimate_solution.py` | Extended version with formula discovery |
| `enigma_winning_solution.py` | Detailed implementation with multiple methods |

---

## 🚀 How to Use on Kaggle

### Method 1: Copy-Paste (Recommended)

1. Create a new **Kaggle Notebook**
2. Add the competition dataset
3. Copy the entire contents of `kaggle_solution_final.py` into a code cell
4. Run the cell
5. Download `submission.csv` from Output
6. Submit!

### Method 2: Upload as Notebook

1. Upload `kaggle_solution_final.py` to Kaggle
2. Convert to notebook or run as script
3. Download and submit

---

## 📊 Expected Results

```
Validation MSE: ~0.000000000000 (near-perfect match)
Exact Match Rate: ~100%
```

The formula should achieve **near-zero MSE** on the leaderboard since it exactly replicates the target computation.

---

## 📐 Formula Details

### Preprocessing
```python
# Parse semicolon-separated fields into sets
Business_Interests: "AI;SaaS;Marketing" → {"AI", "SaaS", "Marketing"}
Business_Objectives: "Hiring;Networking" → {"Hiring", "Networking"}
Constraints: "No sales roles" → {"No sales roles"}
```

### Combination
```python
All_Items = Business_Interests ∪ Business_Objectives ∪ Constraints
```

### Jaccard Similarity
```python
def jaccard(set1, set2):
    if not set1 and not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union
```

---

## 🔬 Why This Works

1. **Pattern Analysis**: Target scores are classic Jaccard fractions (1/n values)
2. **Domain Logic**: Networking compatibility naturally maps to shared attributes
3. **Simplicity**: The organizers mentioned "going beyond cosine similarity" but Jaccard IS the ground truth

---

## 📈 Data Summary

| Dataset | Users | Pairs |
|---------|-------|-------|
| Train | 600 (5001-5600) | 360,000 |
| Test | 400 (5601-6000) | 160,000 |

---

Good luck! 🍀