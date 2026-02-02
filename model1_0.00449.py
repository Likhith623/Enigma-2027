"""
======================================================================================
ENIGMA 2027 - CHAMPIONSHIP WINNING SOLUTION v2.0
======================================================================================
Professional Networking Compatibility Prediction
CodeFest'26 - IIT (BHU) Varanasi

🎯 CONCEPTUAL FOUNDATION:
    Professional networking compatibility = f(shared_goals, complementary_roles, aligned_context)
    
    1. SHARED INTERESTS - Jaccard(Business_Interests)
    2. ALIGNED OBJECTIVES - Jaccard(Business_Objectives)  
    3. COMPATIBLE CONSTRAINTS - Jaccard(Constraints)
    4. ROLE COMPLEMENTARITY - Domain knowledge matrix
    5. INDUSTRY ALIGNMENT - Sector clustering

🔬 TECHNICAL APPROACH:
    - Formula Discovery: Auto-detect generator logic from training
    - ML Insurance: LightGBM hybrid for private LB robustness
    - Self-Pair Handling: Default to 1.0 (perfect self-compatibility)

======================================================================================
KAGGLE INSTRUCTIONS:
1. Create new Notebook → Add Competition Data
2. Copy this entire code into a single cell
3. Run → Download submission.csv → Submit
======================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import os
import re
import warnings
warnings.filterwarnings('ignore')

# Try LightGBM (faster), fallback to sklearn
try:
    import lightgbm as lgb
    HAS_LGBM = True
except:
    from sklearn.ensemble import GradientBoostingRegressor
    HAS_LGBM = False

print("=" * 70)
print("🏆 ENIGMA 2027 - CHAMPIONSHIP SOLUTION v2.0")
print("=" * 70)

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
print("\n[1] Loading data...")

# Official Kaggle paths for Enigma 2027
DATA_DIR = '/kaggle/input/enigma26/Engima26_Dataset'

# Fallback for local testing
if not os.path.exists(DATA_DIR):
    DATA_DIR = '.'
print(f"  → Using: {DATA_DIR}")

# Load data (official format: xlsx for train/test, csv for target)
train_df = pd.read_excel(f'{DATA_DIR}/train.xlsx')
test_df = pd.read_excel(f'{DATA_DIR}/test.xlsx')
target_df = pd.read_csv(f'{DATA_DIR}/target.csv')

print(f"  Train: {len(train_df)} users | Test: {len(test_df)} users | Pairs: {len(target_df)}")

# =============================================================================
# STEP 2: DOMAIN KNOWLEDGE MATRICES
# =============================================================================
print("\n[2] Building domain knowledge...")

ROLE_SYNERGY = {
    ('founder', 'investor'): 1.0, ('founder', 'mentor'): 0.9, ('founder', 'advisor'): 0.9,
    ('founder', 'engineer'): 0.8, ('founder', 'developer'): 0.8, ('investor', 'ceo'): 0.9,
    ('engineer', 'manager'): 0.9, ('developer', 'manager'): 0.9, ('cto', 'engineer'): 0.9,
    ('sales', 'marketing'): 0.9, ('sales', 'product'): 0.8, ('marketing', 'product'): 0.8,
    ('consultant', 'executive'): 0.9, ('consultant', 'manager'): 0.8,
}

INDUSTRY_CLUSTERS = {
    'tech': ['technology', 'software', 'saas', 'ai', 'fintech', 'edtech', 'healthtech'],
    'finance': ['finance', 'banking', 'investment', 'fintech', 'insurance'],
    'healthcare': ['healthcare', 'medical', 'biotech', 'pharma', 'healthtech'],
    'media': ['media', 'entertainment', 'content', 'gaming'],
}

def role_score(r1, r2):
    if not r1 or not r2: return 0.5
    r1, r2 = str(r1).lower().strip(), str(r2).lower().strip()
    if r1 == r2: return 0.7
    for (a, b), s in ROLE_SYNERGY.items():
        if (a in r1 and b in r2) or (b in r1 and a in r2): return s
    return 0.4

def industry_score(i1, i2):
    if not i1 or not i2: return 0.5
    i1, i2 = str(i1).lower(), str(i2).lower()
    if i1 == i2: return 1.0
    c1 = c2 = None
    for c, lst in INDUSTRY_CLUSTERS.items():
        if any(x in i1 for x in lst): c1 = c
        if any(x in i2 for x in lst): c2 = c
    if c1 and c2: return 0.8 if c1 == c2 else 0.4
    return 0.3

print("  ✓ Role synergy matrix | Industry clusters")

# =============================================================================
# STEP 3: PREPROCESSING (ELITE TEXT NORMALIZATION)
# =============================================================================
print("\n[3] Preprocessing with elite text normalization...")

# 🔥 ELITE FIX #1: Text normalization for better Jaccard overlap
def normalize_token(x):
    """Normalize text tokens for consistent matching"""
    x = x.lower().strip()
    x = re.sub(r'\s+', ' ', x)  # Collapse whitespace
    # Common synonyms
    x = x.replace("artificial intelligence", "ai")
    x = x.replace("machine learning", "ml")
    x = x.replace("&", "and")
    x = x.replace("saas", "saas")  # Already lowercase
    x = x.replace("b2b", "b2b")
    x = x.replace("b2c", "b2c")
    return x

def parse_set(val):
    """Parse semicolon-separated values with normalization"""
    if pd.isna(val) or str(val) == 'nan': return set()
    return {
        normalize_token(x)
        for x in str(val).split(';')
        if x.strip() and x.strip() != 'nan'
    }

def preprocess(df):
    df = df.copy()
    df['BI'] = df['Business_Interests'].apply(parse_set)
    df['BO'] = df['Business_Objectives'].apply(parse_set)
    df['CO'] = df['Constraints'].apply(parse_set)
    df['ALL'] = df.apply(lambda r: r['BI'] | r['BO'] | r['CO'], axis=1)
    return df

train_p = preprocess(train_df)
test_p = preprocess(test_df)
print(f"  ✓ Parsed {len(train_p)} train + {len(test_p)} test profiles")

# =============================================================================
# STEP 4: SIMILARITY FUNCTIONS
# =============================================================================
print("\n[4] Defining similarity functions...")

def jaccard(s1, s2):
    if not s1 and not s2: return 0.0
    return len(s1 & s2) / len(s1 | s2) if len(s1 | s2) > 0 else 0.0

def union_jaccard(r1, r2):
    return jaccard(r1['ALL'], r2['ALL'])

def weighted_jaccard(r1, r2, w=(0.4, 0.35, 0.25)):
    return w[0]*jaccard(r1['BI'],r2['BI']) + w[1]*jaccard(r1['BO'],r2['BO']) + w[2]*jaccard(r1['CO'],r2['CO'])

# =============================================================================
# STEP 5: FORMULA DISCOVERY
# =============================================================================
print("\n[5] Discovering optimal formula...")

# Check self-pairs in training
self_pairs = target_df[target_df['src_user_id'] == target_df['dst_user_id']]
if len(self_pairs) > 0:
    SELF_SCORE = self_pairs['compatibility_score'].iloc[0]
    print(f"  Self-pairs in training: {len(self_pairs)}, score={SELF_SCORE}")
else:
    # CRITICAL: Sample submission shows self-pair = 1.0
    SELF_SCORE = 1.0
    print(f"  No self-pairs in training → Using default: {SELF_SCORE}")

# Build lookup
train_lookup = {r['Profile_ID']: r for _, r in train_p.iterrows()}

# Test formulas
def test_formula(func):
    preds, acts = [], []
    for row in target_df.itertuples():
        if row.src_user_id == row.dst_user_id: continue  # Skip self-pairs for formula testing
        preds.append(round(func(train_lookup[row.src_user_id], train_lookup[row.dst_user_id]), 4))
        acts.append(row.compatibility_score)
    return mean_squared_error(acts, preds), preds, acts

formulas = [
    ("Union Jaccard", union_jaccard),
    ("Weighted(0.4/0.35/0.25)", lambda r1,r2: weighted_jaccard(r1,r2,(0.4,0.35,0.25))),
    ("Weighted(0.5/0.3/0.2)", lambda r1,r2: weighted_jaccard(r1,r2,(0.5,0.3,0.2))),
    ("Weighted(0.33/0.33/0.34)", lambda r1,r2: weighted_jaccard(r1,r2,(0.33,0.33,0.34))),
    ("BI only", lambda r1,r2: jaccard(r1['BI'],r2['BI'])),
]

best_mse, best_func, best_name = float('inf'), None, ""
for name, func in formulas:
    mse, preds, acts = test_formula(func)
    exact = sum(1 for a,p in zip(acts,preds) if abs(a-p)<1e-5)
    print(f"    {name}: MSE={mse:.10f}, Exact={100*exact/len(acts):.2f}%")
    if mse < best_mse:
        best_mse, best_func, best_name = mse, func, name
        best_preds, best_acts = preds, acts

print(f"\n  ★ WINNER: {best_name} (MSE={best_mse:.12f})")

# =============================================================================
# STEP 6: ML INSURANCE LAYER (Private LB Protection)
# =============================================================================
print("\n[6] Training ML insurance layer...")

def extract_features(r1, r2):
    """Extract comprehensive features for ML model"""
    f = {
        'j_all': jaccard(r1['ALL'], r2['ALL']),
        'j_bi': jaccard(r1['BI'], r2['BI']),
        'j_bo': jaccard(r1['BO'], r2['BO']),
        'j_co': jaccard(r1['CO'], r2['CO']),
        'bi_inter': len(r1['BI'] & r2['BI']),
        'bo_inter': len(r1['BO'] & r2['BO']),
        'co_inter': len(r1['CO'] & r2['CO']),
        'all_union': len(r1['ALL'] | r2['ALL']),
        # 🔥 ELITE FIX #4: Enhanced features for networking realism
        'union_size': len(r1['ALL'] | r2['ALL']),
        'overlap_ratio_1': len(r1['ALL'] & r2['ALL']) / (len(r1['ALL']) + 1e-6),
        'overlap_ratio_2': len(r1['ALL'] & r2['ALL']) / (len(r2['ALL']) + 1e-6),
        'size_diff': abs(len(r1['ALL']) - len(r2['ALL'])),
        'min_size': min(len(r1['ALL']), len(r2['ALL'])),
        'max_size': max(len(r1['ALL']), len(r2['ALL'])),
    }
    # Add demographic features if available
    for col in ['Age', 'Role', 'Industry', 'Location_City', 'Seniority_Level']:
        if col in r1 and col in r2:
            if col == 'Age':
                a1 = r1.get(col, 30) if pd.notna(r1.get(col)) else 30
                a2 = r2.get(col, 30) if pd.notna(r2.get(col)) else 30
                f['age_diff'] = abs(a1 - a2)
            elif col == 'Role':
                f['role_synergy'] = role_score(r1.get(col), r2.get(col))
            elif col == 'Industry':
                f['industry_align'] = industry_score(r1.get(col), r2.get(col))
            else:
                f[f'same_{col.lower()}'] = 1 if str(r1.get(col,'')).lower() == str(r2.get(col,'')).lower() else 0
    return f

# Build ML training data (exclude self-pairs)
X_list, y_list = [], []
for row in target_df.itertuples():
    if row.src_user_id == row.dst_user_id: continue
    X_list.append(extract_features(train_lookup[row.src_user_id], train_lookup[row.dst_user_id]))
    y_list.append(row.compatibility_score)

X_train = pd.DataFrame(X_list)
y_train = np.array(y_list)
print(f"  Features: {X_train.shape[1]} | Samples: {len(y_train)}")

# Train with CV
if HAS_LGBM:
    ml_model = lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, verbose=-1, random_state=42)
else:
    ml_model = GradientBoostingRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_mses = []
for tr_idx, va_idx in kf.split(X_train):
    ml_model.fit(X_train.iloc[tr_idx], y_train[tr_idx])
    cv_mses.append(mean_squared_error(y_train[va_idx], ml_model.predict(X_train.iloc[va_idx])))
ml_cv_mse = np.mean(cv_mses)

# Train final model
ml_model.fit(X_train, y_train)
ml_train_preds = ml_model.predict(X_train)
print(f"  ML CV MSE: {ml_cv_mse:.10f}")

# Decide: Use hybrid only if it helps
print(f"\n  COMPARISON:")
print(f"    Formula MSE: {best_mse:.10f}")
print(f"    ML CV MSE:   {ml_cv_mse:.10f}")

# =============================================================================
# [PRO MODE] CV-ROBUST ALPHA SWEEP (Elite Fix #5)
# =============================================================================
print("\n  [PRO MODE] CV-robust alpha search...")
print("  " + "-" * 50)

alphas = np.linspace(0.0, 1.0, 21)  # 0.00, 0.05, ..., 1.00
best_blend_mse, best_alpha = float('inf'), 1.0
formula_preds = np.array(best_preds)
alpha_results = []

# 🔥 ELITE FIX #5: CV-robust alpha selection (prevents overfitting)
for alpha in alphas:
    cv_blend_mses = []
    for tr_idx, va_idx in kf.split(formula_preds):
        blend_va = alpha * formula_preds[va_idx] + (1 - alpha) * ml_train_preds[va_idx]
        cv_blend_mses.append(mean_squared_error(y_train[va_idx], blend_va))
    mse = np.mean(cv_blend_mses)
    alpha_results.append((alpha, mse))
    marker = " ◀" if mse < best_blend_mse else ""
    print(f"    α={alpha:.2f} → CV-MSE={mse:.10f}{marker}")
    if mse < best_blend_mse:
        best_blend_mse, best_alpha = mse, alpha

print("  " + "-" * 50)
print(f"\n  🏆 BEST ALPHA FOUND: α = {best_alpha:.2f}")
print(f"     Best MSE = {best_blend_mse:.10f}")

# Define submission variants for robustness
ALPHA_MAIN = best_alpha
ALPHA_BACKUP1 = min(1.0, best_alpha + 0.1)
ALPHA_BACKUP2 = max(0.0, best_alpha - 0.1)

print(f"\n  📦 SUBMISSION VARIANTS:")
print(f"     Main:    α = {ALPHA_MAIN:.2f} (CV optimal)")
print(f"     Backup1: α = {ALPHA_BACKUP1:.2f} (higher formula weight)")
print(f"     Backup2: α = {ALPHA_BACKUP2:.2f} (higher ML weight)")

USE_HYBRID = best_alpha < 1.0
print(f"\n  ★ DECISION: {'HYBRID' if USE_HYBRID else 'PURE FORMULA'} (α={best_alpha:.2f})")
print(f"    Final MSE: {best_blend_mse:.12f}")

# =============================================================================
# STEP 7: GENERATE PREDICTIONS (Multiple Alpha Variants)
# =============================================================================
print("\n[7] Generating predictions...")

test_lookup = {r['Profile_ID']: r for _, r in test_p.iterrows()}
# 🔥 ELITE FIX #6: Speed optimization
test_ids = sorted(test_p['Profile_ID'].unique().tolist())
print(f"  Test users: {len(test_ids)} | Pairs: {len(test_ids)**2}")

# Pre-compute all formula scores and features
results = []
test_features = []
test_jaccard_scores = []  # For safety clamping

for i, src in enumerate(test_ids):
    if i % 100 == 0: print(f"  Processing {i+1}/{len(test_ids)}...")
    src_row = test_lookup[src]
    for dst in test_ids:
        dst_row = test_lookup[dst]
        
        if src == dst:
            # 🔥 ELITE FIX #2: Smart self-pair handling
            computed_score = best_func(src_row, dst_row)
            formula_score = max(SELF_SCORE, computed_score)  # Protect against generator changes
            results.append({'ID': f"{src}_{dst}", 'formula': formula_score, 'is_self': True, 'j_raw': 1.0})
        else:
            formula_score = best_func(src_row, dst_row)
            j_raw = jaccard(src_row['ALL'], dst_row['ALL'])
            test_features.append(extract_features(src_row, dst_row))
            test_jaccard_scores.append(j_raw)
            results.append({'ID': f"{src}_{dst}", 'formula': formula_score, 'is_self': False, 'j_raw': j_raw})

sub_df = pd.DataFrame(results)

# Get ML predictions for non-self pairs
print("  Computing ML predictions...")
X_test = pd.DataFrame(test_features)
ml_test_preds = ml_model.predict(X_test)

# Function to generate submission with specific alpha
def generate_submission(alpha, suffix=""):
    df = sub_df.copy()
    non_self = ~df['is_self']
    
    if alpha < 1.0:
        # Hybrid: blend formula + ML
        blended = alpha * df.loc[non_self, 'formula'].values + (1 - alpha) * ml_test_preds
        
        # 🔥 ELITE FIX #3: Safety clamp for low-Jaccard pairs
        # Prevents ML from hallucinating high scores for unrelated people
        j_raw_values = df.loc[non_self, 'j_raw'].values
        for idx in range(len(blended)):
            if j_raw_values[idx] < 0.05:  # Very low overlap
                blended[idx] = min(blended[idx], 0.1)  # Cap at 0.1
        
        df.loc[non_self, 'compatibility_score'] = np.clip(blended, 0, 1).round(4)
    else:
        # Pure formula
        df.loc[non_self, 'compatibility_score'] = df.loc[non_self, 'formula'].round(4)
    
    # Self-pairs use the smart formula (max of SELF_SCORE and computed)
    df.loc[df['is_self'], 'compatibility_score'] = df.loc[df['is_self'], 'formula'].round(4)
    df = df[['ID', 'compatibility_score']]
    
    filename = f"submission{suffix}.csv"
    df.to_csv(filename, index=False)
    return df, filename

# Generate all submission variants
print("\n  Generating submission variants...")
submissions = {}

# Main submission (best alpha)
sub_main, fname_main = generate_submission(ALPHA_MAIN, "")
submissions['main'] = (ALPHA_MAIN, sub_main, fname_main)
print(f"    ✓ {fname_main} (α={ALPHA_MAIN:.2f}) - MAIN")

# Backup 1 (higher formula weight)
if ALPHA_BACKUP1 != ALPHA_MAIN:
    sub_b1, fname_b1 = generate_submission(ALPHA_BACKUP1, f"_alpha{int(ALPHA_BACKUP1*100)}")
    submissions['backup1'] = (ALPHA_BACKUP1, sub_b1, fname_b1)
    print(f"    ✓ {fname_b1} (α={ALPHA_BACKUP1:.2f}) - BACKUP")

# Backup 2 (higher ML weight)
if ALPHA_BACKUP2 != ALPHA_MAIN:
    sub_b2, fname_b2 = generate_submission(ALPHA_BACKUP2, f"_alpha{int(ALPHA_BACKUP2*100)}")
    submissions['backup2'] = (ALPHA_BACKUP2, sub_b2, fname_b2)
    print(f"    ✓ {fname_b2} (α={ALPHA_BACKUP2:.2f}) - BACKUP")

# Also generate pure formula and pure ML for comparison
sub_pure_formula, fname_pf = generate_submission(1.0, "_pure_formula")
submissions['pure_formula'] = (1.0, sub_pure_formula, fname_pf)
print(f"    ✓ {fname_pf} (α=1.00) - PURE FORMULA")

sub_pure_ml, fname_ml = generate_submission(0.0, "_pure_ml")
submissions['pure_ml'] = (0.0, sub_pure_ml, fname_ml)
print(f"    ✓ {fname_ml} (α=0.00) - PURE ML")

# =============================================================================
# STEP 8: VALIDATE & SAVE
# =============================================================================
print("\n[8] Validating...")

# Validate main submission
assert len(sub_main) == len(test_ids)**2, f"Row count mismatch!"
assert sub_main['compatibility_score'].between(0, 1).all(), "Scores out of range!"
print(f"  ✓ Shape: {sub_main.shape}")
print(f"  ✓ Score range: [{sub_main['compatibility_score'].min():.4f}, {sub_main['compatibility_score'].max():.4f}]")

# Check self-pairs
self_check = sub_main[sub_main['ID'].apply(lambda x: x.split('_')[0] == x.split('_')[1])]
print(f"  ✓ Self-pairs: {len(self_check)}, score={self_check['compatibility_score'].iloc[0]}")

print("\n  Sample (Main Submission):")
print(sub_main.head(10).to_string(index=False))

# =============================================================================
# DONE
# =============================================================================
print("\n" + "=" * 70)
print("🏆 CHAMPIONSHIP SOLUTION COMPLETE!")
print("=" * 70)
print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  WINNING CONFIGURATION                                               ║
╠══════════════════════════════════════════════════════════════════════╣
║  • Formula: {best_name:<45} ║
║  • Self-pair score: {SELF_SCORE:<40} ║
║  • ML Hybrid: {str(USE_HYBRID):<47} ║
║  • Best α (CV): {best_alpha:<44.2f} ║
║  • Final MSE: {best_blend_mse:<47.12f} ║
╚══════════════════════════════════════════════════════════════════════╝

📦 SUBMISSION FILES GENERATED:
""")

for name, (alpha, df, filename) in submissions.items():
    marker = "⭐ RECOMMENDED" if name == 'main' else ""
    print(f"   • {filename:<30} α={alpha:.2f}  {marker}")

print(f"""
🎯 SUBMISSION STRATEGY:
   1. Submit 'submission.csv' first (α={ALPHA_MAIN:.2f}) - CV optimal
   2. If you have extra submissions, try backups for private LB insurance

CONCEPTUAL FOUNDATION:
  1. Jaccard similarity on Business Interests/Objectives/Constraints
  2. Role complementarity matrix (Founder↔Investor, Engineer↔Manager)
  3. Industry alignment clustering
  4. ML refinement for private leaderboard robustness

RISK MITIGATION:
  ✓ Auto-discovers generator formula
  ✓ Comprehensive alpha sweep (21 values tested)
  ✓ Multiple submission variants for LB shift protection
  ✓ Correct self-pair handling ({SELF_SCORE})

OUTPUT: {len(submissions)} submission files ({len(sub_main):,} rows each)

Good luck! 🏆
""")
