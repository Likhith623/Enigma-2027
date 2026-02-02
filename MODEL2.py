"""
======================================================================================
ENIGMA 2027 - ULTIMATE SOLUTION v5.0 (FORMULA HUNTER)
======================================================================================
🎯 DISCOVER THE EXACT GENERATOR FORMULA 🎯

PHILOSOPHY:
  "The data is synthetically generated. If we find the EXACT formula,
   we achieve MSE ≈ 0. This solution systematically hunts for it."

APPROACH:
  1. Test ALL possible formula combinations (Jaccard, Dice, Overlap, Weighted)
  2. Use Isotonic Regression to learn the EXACT transformation
  3. Explore weighted column combinations (BI, BO, CO with different weights)
  4. Check for asymmetry in the formula
  5. Use polynomial fitting to capture any nonlinearity perfectly

======================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import PolynomialFeatures
import os
import re
import warnings
from itertools import product
warnings.filterwarnings('ignore')

print("=" * 70)
print("🎯 ENIGMA 2027 - ULTIMATE SOLUTION v5.0 (FORMULA HUNTER) 🎯")
print("=" * 70)
print("\n  Goal: Discover the EXACT generator formula for MSE → 0")

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
print("\n[1] Loading data...")

DATA_DIR = '/kaggle/input/enigma26/Engima26_Dataset'
if not os.path.exists(DATA_DIR):
    DATA_DIR = '.'
print(f"  → Using: {DATA_DIR}")

train_df = pd.read_excel(f'{DATA_DIR}/train.xlsx')
test_df = pd.read_excel(f'{DATA_DIR}/test.xlsx')
target_df = pd.read_csv(f'{DATA_DIR}/target.csv')

print(f"  Train: {len(train_df)} users | Test: {len(test_df)} users | Pairs: {len(target_df)}")

# =============================================================================
# STEP 2: TEXT NORMALIZATION
# =============================================================================
print("\n[2] Text normalization...")

def normalize_token(x):
    """Comprehensive synonym normalization"""
    x = x.lower().strip()
    x = re.sub(r'\s+', ' ', x)
    x = re.sub(r'[^\w\s]', '', x)
    
    synonyms = {
        'artificial intelligence': 'ai', 'machine learning': 'ml',
        'deep learning': 'dl', 'natural language processing': 'nlp',
        'computer vision': 'cv', 'data science': 'ds',
        'software as a service': 'saas', 'platform as a service': 'paas',
        'infrastructure as a service': 'iaas',
        'business to business': 'b2b', 'business to consumer': 'b2c',
        'research and development': 'rd', 'mergers and acquisitions': 'ma',
        'chief executive officer': 'ceo', 'chief technology officer': 'cto',
        'chief financial officer': 'cfo', 'chief operating officer': 'coo',
        'venture capital': 'vc', 'private equity': 'pe',
        'initial public offering': 'ipo',
        'user experience': 'ux', 'user interface': 'ui',
        'product management': 'pm', 'project management': 'pm',
        'human resources': 'hr', 'information technology': 'it',
    }
    for k, v in synonyms.items():
        x = x.replace(k, v)
    x = x.replace('&', 'and')
    return x

def parse_set(val):
    if pd.isna(val) or str(val) == 'nan':
        return set()
    return {normalize_token(t) for t in str(val).split(';') if t.strip() and t.strip() != 'nan'}

def preprocess(df):
    df = df.copy()
    df['BI'] = df['Business_Interests'].apply(parse_set)
    df['BO'] = df['Business_Objectives'].apply(parse_set)
    df['CO'] = df['Constraints'].apply(parse_set)
    df['ALL'] = df.apply(lambda r: r['BI'] | r['BO'] | r['CO'], axis=1)
    df['BI_BO'] = df.apply(lambda r: r['BI'] | r['BO'], axis=1)
    return df

train_p = preprocess(train_df)
test_p = preprocess(test_df)
print(f"  ✓ Parsed {len(train_p)} train + {len(test_p)} test profiles")

# =============================================================================
# STEP 3: ALL SIMILARITY FUNCTIONS
# =============================================================================
print("\n[3] Defining similarity functions...")

def jaccard(s1, s2):
    if not s1 and not s2: return 0.0
    union = s1 | s2
    if len(union) == 0: return 0.0
    return len(s1 & s2) / len(union)

def dice(s1, s2):
    if not s1 and not s2: return 0.0
    total = len(s1) + len(s2)
    if total == 0: return 0.0
    return 2 * len(s1 & s2) / total

def overlap(s1, s2):
    if not s1 or not s2: return 0.0
    min_size = min(len(s1), len(s2))
    if min_size == 0: return 0.0
    return len(s1 & s2) / min_size

def cosine_set(s1, s2):
    if not s1 or not s2: return 0.0
    prod = len(s1) * len(s2)
    if prod == 0: return 0.0
    return len(s1 & s2) / np.sqrt(prod)

print("  ✓ Jaccard, Dice, Overlap, Cosine defined")

# =============================================================================
# STEP 4: DISCOVER SELF-PAIR SCORE
# =============================================================================
print("\n[4] Discovering self-pair pattern...")

self_pairs = target_df[target_df['src_user_id'] == target_df['dst_user_id']]
if len(self_pairs) > 0:
    SELF_SCORE = self_pairs['compatibility_score'].iloc[0]
    print(f"  Found {len(self_pairs)} self-pairs, score={SELF_SCORE}")
else:
    SELF_SCORE = 1.0
    print(f"  No self-pairs found, assuming SELF_SCORE = {SELF_SCORE}")

# =============================================================================
# STEP 5: MASSIVE FORMULA SEARCH
# =============================================================================
print("\n[5] 🔍 MASSIVE FORMULA SEARCH...")

train_lookup = {r['Profile_ID']: r for _, r in train_p.iterrows()}
train_pairs = target_df[target_df['src_user_id'] != target_df['dst_user_id']].copy()
print(f"  Training pairs (non-self): {len(train_pairs)}")

# Precompute all component similarities
print("  Precomputing all similarities...")
all_sims = []
y_train = []
groups = []

for row in train_pairs.itertuples():
    r1 = train_lookup[row.src_user_id]
    r2 = train_lookup[row.dst_user_id]
    
    sims = {
        # Jaccard variants
        'j_all': jaccard(r1['ALL'], r2['ALL']),
        'j_bi': jaccard(r1['BI'], r2['BI']),
        'j_bo': jaccard(r1['BO'], r2['BO']),
        'j_co': jaccard(r1['CO'], r2['CO']),
        'j_bi_bo': jaccard(r1['BI_BO'], r2['BI_BO']),
        
        # Dice variants
        'd_all': dice(r1['ALL'], r2['ALL']),
        'd_bi': dice(r1['BI'], r2['BI']),
        'd_bo': dice(r1['BO'], r2['BO']),
        'd_co': dice(r1['CO'], r2['CO']),
        
        # Overlap variants
        'o_all': overlap(r1['ALL'], r2['ALL']),
        'o_bi': overlap(r1['BI'], r2['BI']),
        'o_bo': overlap(r1['BO'], r2['BO']),
        'o_co': overlap(r1['CO'], r2['CO']),
        
        # Cosine variants
        'c_all': cosine_set(r1['ALL'], r2['ALL']),
        'c_bi': cosine_set(r1['BI'], r2['BI']),
        
        # Intersection counts
        'inter_all': len(r1['ALL'] & r2['ALL']),
        'inter_bi': len(r1['BI'] & r2['BI']),
        'inter_bo': len(r1['BO'] & r2['BO']),
        'inter_co': len(r1['CO'] & r2['CO']),
        
        # Union counts
        'union_all': len(r1['ALL'] | r2['ALL']),
        'union_bi': len(r1['BI'] | r2['BI']),
        
        # Size features
        'size_1': len(r1['ALL']),
        'size_2': len(r2['ALL']),
    }
    all_sims.append(sims)
    y_train.append(row.compatibility_score)
    pair_id = f"{min(row.src_user_id, row.dst_user_id)}_{max(row.src_user_id, row.dst_user_id)}"
    groups.append(pair_id)

sim_df = pd.DataFrame(all_sims)
y_train = np.array(y_train)
groups = np.array(groups)

# Test all single-column formulas
print("\n  Testing single-column formulas...")
formula_results = []

for col in sim_df.columns:
    if col.startswith('size') or col.startswith('inter') or col.startswith('union'):
        continue
    preds = sim_df[col].values
    mse = mean_squared_error(y_train, preds)
    formula_results.append((col, mse, preds))
    print(f"    {col}: MSE={mse:.10f}")

# Sort by MSE
formula_results.sort(key=lambda x: x[1])
print(f"\n  🏆 BEST single formula: {formula_results[0][0]} (MSE={formula_results[0][1]:.10f})")

best_base_col = formula_results[0][0]
best_base_preds = formula_results[0][2]
best_base_mse = formula_results[0][1]

# =============================================================================
# STEP 6: WEIGHTED COMBINATION SEARCH
# =============================================================================
print("\n[6] 🔍 WEIGHTED COMBINATION SEARCH...")

# Try weighted combinations of BI, BO, CO Jaccards
print("  Testing weighted Jaccard combinations (w_bi + w_bo + w_co = 1)...")

best_weighted_mse = float('inf')
best_weights = None

for w_bi in np.linspace(0, 1, 11):
    for w_bo in np.linspace(0, 1 - w_bi, 11):
        w_co = 1 - w_bi - w_bo
        if w_co < 0:
            continue
        
        weighted = w_bi * sim_df['j_bi'] + w_bo * sim_df['j_bo'] + w_co * sim_df['j_co']
        mse = mean_squared_error(y_train, weighted)
        
        if mse < best_weighted_mse:
            best_weighted_mse = mse
            best_weights = (w_bi, w_bo, w_co)

if best_weights:
    print(f"  Best weighted Jaccard: w_bi={best_weights[0]:.2f}, w_bo={best_weights[1]:.2f}, w_co={best_weights[2]:.2f}")
    print(f"  MSE: {best_weighted_mse:.10f}")

# Compare with Union Jaccard
if best_weighted_mse < best_base_mse:
    print(f"  → Weighted combination BEATS {best_base_col}!")
    best_base_preds = best_weights[0] * sim_df['j_bi'] + best_weights[1] * sim_df['j_bo'] + best_weights[2] * sim_df['j_co']
    best_base_mse = best_weighted_mse
else:
    print(f"  → {best_base_col} remains best base")

# =============================================================================
# STEP 7: ISOTONIC REGRESSION (LEARN EXACT TRANSFORMATION)
# =============================================================================
print("\n[7] 🔍 ISOTONIC REGRESSION (Learn exact transformation)...")

# Isotonic regression learns a monotonic function that perfectly maps base → target
iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_preds = iso_reg.fit_transform(best_base_preds, y_train)
iso_mse = mean_squared_error(y_train, iso_preds)
print(f"  Isotonic MSE (train): {iso_mse:.10f}")

# But we need CV to avoid overfitting
print("  Cross-validating isotonic regression...")
gkf = GroupKFold(n_splits=5)
iso_cv_preds = np.zeros(len(y_train))

for tr_idx, va_idx in gkf.split(best_base_preds, groups=groups):
    iso_cv = IsotonicRegression(out_of_bounds='clip')
    iso_cv.fit(best_base_preds[tr_idx], y_train[tr_idx])
    iso_cv_preds[va_idx] = iso_cv.predict(best_base_preds[va_idx])

iso_cv_mse = mean_squared_error(y_train, iso_cv_preds)
print(f"  Isotonic MSE (CV): {iso_cv_mse:.10f}")

# =============================================================================
# STEP 8: POLYNOMIAL TRANSFORMATION
# =============================================================================
print("\n[8] 🔍 POLYNOMIAL TRANSFORMATION SEARCH...")

best_poly_mse = float('inf')
best_poly_degree = 1
best_poly_model = None

for degree in range(1, 8):
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly.fit_transform(best_base_preds.reshape(-1, 1))
    
    # CV
    poly_cv_preds = np.zeros(len(y_train))
    for tr_idx, va_idx in gkf.split(X_poly, groups=groups):
        lr = LinearRegression()
        lr.fit(X_poly[tr_idx], y_train[tr_idx])
        poly_cv_preds[va_idx] = lr.predict(X_poly[va_idx])
    
    poly_cv_preds = np.clip(poly_cv_preds, 0, 1)
    mse = mean_squared_error(y_train, poly_cv_preds)
    print(f"    Degree {degree}: CV MSE={mse:.10f}")
    
    if mse < best_poly_mse:
        best_poly_mse = mse
        best_poly_degree = degree

print(f"  Best polynomial degree: {best_poly_degree} (MSE={best_poly_mse:.10f})")

# =============================================================================
# STEP 9: MULTI-FEATURE LINEAR COMBINATION
# =============================================================================
print("\n[9] 🔍 MULTI-FEATURE LINEAR COMBINATION...")

# Use Ridge regression on all similarity features
feature_cols = [c for c in sim_df.columns if not c.startswith('size') and not c.startswith('union')]
X_features = sim_df[feature_cols].values

# CV
ridge_cv_preds = np.zeros(len(y_train))
for tr_idx, va_idx in gkf.split(X_features, groups=groups):
    ridge = Ridge(alpha=0.1)
    ridge.fit(X_features[tr_idx], y_train[tr_idx])
    ridge_cv_preds[va_idx] = ridge.predict(X_features[va_idx])

ridge_cv_preds = np.clip(ridge_cv_preds, 0, 1)
ridge_cv_mse = mean_squared_error(y_train, ridge_cv_preds)
print(f"  Ridge CV MSE: {ridge_cv_mse:.10f}")

# Show coefficients
ridge_full = Ridge(alpha=0.1)
ridge_full.fit(X_features, y_train)
print("\n  Feature importance (Ridge coefficients):")
coef_df = pd.DataFrame({'feature': feature_cols, 'coef': ridge_full.coef_})
coef_df = coef_df.reindex(coef_df['coef'].abs().sort_values(ascending=False).index)
for _, row in coef_df.head(10).iterrows():
    print(f"    {row['feature']}: {row['coef']:.6f}")

# =============================================================================
# STEP 10: GRADIENT BOOSTING RESIDUAL
# =============================================================================
print("\n[10] 🔍 GRADIENT BOOSTING RESIDUAL CORRECTION...")

# Use the best base predictions
if iso_cv_mse < best_base_mse and iso_cv_mse < best_poly_mse:
    formula_preds = iso_cv_preds.copy()
    print(f"  Using ISOTONIC as base (MSE={iso_cv_mse:.10f})")
elif best_poly_mse < best_base_mse:
    # Recompute polynomial predictions
    poly = PolynomialFeatures(degree=best_poly_degree, include_bias=True)
    X_poly = poly.fit_transform(best_base_preds.reshape(-1, 1))
    lr = LinearRegression()
    lr.fit(X_poly, y_train)
    formula_preds = np.clip(lr.predict(X_poly), 0, 1)
    print(f"  Using POLYNOMIAL (degree={best_poly_degree}) as base")
else:
    formula_preds = best_base_preds.copy()
    print(f"  Using RAW {best_base_col} as base (MSE={best_base_mse:.10f})")

# Compute residuals
residuals = y_train - formula_preds

# Feature engineering for residual
X_residual = sim_df.copy()
X_residual['formula_pred'] = formula_preds

# Add stability noise
np.random.seed(42)
X_residual_stable = X_residual + np.random.normal(0, 1e-6, X_residual.shape)

# Train GB residual model
print("\n  Training residual correction model...")
gb = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)

# CV for residuals
oof_residuals = cross_val_predict(gb, X_residual_stable, residuals, cv=gkf, groups=groups)
gb.fit(X_residual_stable, residuals)

# Clamp residuals
RESIDUAL_CLAMP = 0.15
oof_residuals = np.clip(oof_residuals, -RESIDUAL_CLAMP, RESIDUAL_CLAMP)

# Corrected predictions
ml_corrected = formula_preds + oof_residuals
ml_corrected = np.clip(ml_corrected, 0, 1)

residual_mse = mean_squared_error(y_train, ml_corrected)
print(f"  Residual-corrected MSE: {residual_mse:.10f}")

# =============================================================================
# STEP 11: OPTIMAL ALPHA BLENDING
# =============================================================================
print("\n[11] Finding optimal alpha...")

y_minus_f = y_train - formula_preds
ml_minus_f = ml_corrected - formula_preds

cov_num = np.cov(y_minus_f, ml_minus_f)[0, 1]
var_den = np.var(ml_minus_f) + 1e-10
alpha_analytical = np.clip(cov_num / var_den, 0, 1)

# Grid search validation
best_alpha = alpha_analytical
best_mse = float('inf')

for alpha in np.linspace(0, 1, 101):
    blend = alpha * ml_corrected + (1 - alpha) * formula_preds
    mse = mean_squared_error(y_train, blend)
    if mse < best_mse:
        best_mse = mse
        best_alpha = alpha

print(f"  Analytical α: {alpha_analytical:.4f}")
print(f"  Best α: {best_alpha:.4f} (MSE={best_mse:.10f})")

# Final training predictions
final_train_preds = best_alpha * ml_corrected + (1 - best_alpha) * formula_preds
final_train_mse = mean_squared_error(y_train, final_train_preds)
print(f"  Final training MSE: {final_train_mse:.10f}")

# =============================================================================
# STEP 12: ADAPTIVE SAFETY CLAMP
# =============================================================================
print("\n[12] Adaptive safety clamp...")

LOW_J_THRESHOLD = np.percentile(best_base_preds, 10)
low_j_mask = best_base_preds < LOW_J_THRESHOLD
if low_j_mask.sum() > 0:
    LOW_J_MAX_SCORE = np.percentile(y_train[low_j_mask], 95)
    LOW_J_MAX_SCORE = max(LOW_J_MAX_SCORE, 0.05)
else:
    LOW_J_MAX_SCORE = 0.1

print(f"  Threshold: {LOW_J_THRESHOLD:.4f}")
print(f"  Max score for low-J: {LOW_J_MAX_SCORE:.4f}")

safe_train_preds = final_train_preds.copy()
safe_train_preds[low_j_mask] = np.minimum(safe_train_preds[low_j_mask], LOW_J_MAX_SCORE)

safe_mse = mean_squared_error(y_train, safe_train_preds)
USE_SAFETY_CLAMP = safe_mse <= final_train_mse
print(f"  Safety clamp MSE: {safe_mse:.10f} → {'ENABLED' if USE_SAFETY_CLAMP else 'DISABLED'}")

# =============================================================================
# STEP 13: GENERATE TEST PREDICTIONS
# =============================================================================
print("\n[13] Generating test predictions...")

test_lookup = {r['Profile_ID']: r for _, r in test_p.iterrows()}
test_ids = sorted(test_p['Profile_ID'].unique().tolist())
n_test = len(test_ids)
print(f"  Test users: {n_test} | Pairs: {n_test**2}")

# Compute test similarities
results = []
test_sims = []
test_base_preds = []

for i, src in enumerate(test_ids):
    if i % 100 == 0:
        print(f"  Processing {i+1}/{n_test}...")
    
    src_row = test_lookup[src]
    for dst in test_ids:
        dst_row = test_lookup[dst]
        pair_id = f"{src}_{dst}"
        
        if src == dst:
            results.append({'ID': pair_id, 'is_self': True, 'score': SELF_SCORE})
        else:
            sims = {
                'j_all': jaccard(src_row['ALL'], dst_row['ALL']),
                'j_bi': jaccard(src_row['BI'], dst_row['BI']),
                'j_bo': jaccard(src_row['BO'], dst_row['BO']),
                'j_co': jaccard(src_row['CO'], dst_row['CO']),
                'j_bi_bo': jaccard(src_row['BI_BO'], dst_row['BI_BO']),
                'd_all': dice(src_row['ALL'], dst_row['ALL']),
                'd_bi': dice(src_row['BI'], dst_row['BI']),
                'd_bo': dice(src_row['BO'], dst_row['BO']),
                'd_co': dice(src_row['CO'], dst_row['CO']),
                'o_all': overlap(src_row['ALL'], dst_row['ALL']),
                'o_bi': overlap(src_row['BI'], dst_row['BI']),
                'o_bo': overlap(src_row['BO'], dst_row['BO']),
                'o_co': overlap(src_row['CO'], dst_row['CO']),
                'c_all': cosine_set(src_row['ALL'], dst_row['ALL']),
                'c_bi': cosine_set(src_row['BI'], dst_row['BI']),
                'inter_all': len(src_row['ALL'] & dst_row['ALL']),
                'inter_bi': len(src_row['BI'] & dst_row['BI']),
                'inter_bo': len(src_row['BO'] & dst_row['BO']),
                'inter_co': len(src_row['CO'] & dst_row['CO']),
                'union_all': len(src_row['ALL'] | dst_row['ALL']),
                'union_bi': len(src_row['BI'] | dst_row['BI']),
                'size_1': len(src_row['ALL']),
                'size_2': len(dst_row['ALL']),
            }
            test_sims.append(sims)
            
            # Compute base prediction
            if best_weights:
                base = best_weights[0] * sims['j_bi'] + best_weights[1] * sims['j_bo'] + best_weights[2] * sims['j_co']
            else:
                base = sims[best_base_col]
            test_base_preds.append(base)
            
            results.append({'ID': pair_id, 'is_self': False})

test_sim_df = pd.DataFrame(test_sims)
test_base_preds = np.array(test_base_preds)

# Apply same transformation as training
if iso_cv_mse < best_base_mse and iso_cv_mse < best_poly_mse:
    test_formula_preds = iso_reg.predict(test_base_preds)
elif best_poly_mse < best_base_mse:
    poly = PolynomialFeatures(degree=best_poly_degree, include_bias=True)
    X_test_poly = poly.fit_transform(test_base_preds.reshape(-1, 1))
    # Refit on all training data
    X_train_poly = poly.fit_transform(best_base_preds.reshape(-1, 1))
    lr = LinearRegression()
    lr.fit(X_train_poly, y_train)
    test_formula_preds = np.clip(lr.predict(X_test_poly), 0, 1)
else:
    test_formula_preds = test_base_preds

# Add formula prediction to features
test_sim_df['formula_pred'] = test_formula_preds

# Predict residuals
test_residuals = gb.predict(test_sim_df)
test_residuals = np.clip(test_residuals, -RESIDUAL_CLAMP, RESIDUAL_CLAMP)

# Final predictions
test_ml_corrected = test_formula_preds + test_residuals
test_ml_corrected = np.clip(test_ml_corrected, 0, 1)

test_final = best_alpha * test_ml_corrected + (1 - best_alpha) * test_formula_preds
test_final = np.clip(test_final, 0, 1)

# Apply safety clamp
if USE_SAFETY_CLAMP:
    low_j_mask = test_base_preds < LOW_J_THRESHOLD
    test_final[low_j_mask] = np.minimum(test_final[low_j_mask], LOW_J_MAX_SCORE)

# =============================================================================
# STEP 14: CREATE SUBMISSION
# =============================================================================
print("\n[14] Creating submission...")

sub_df = pd.DataFrame(results)

# Fill in non-self scores
idx = 0
for i in range(len(sub_df)):
    if not sub_df.loc[i, 'is_self']:
        sub_df.loc[i, 'score'] = test_final[idx]
        idx += 1

sub_df['compatibility_score'] = sub_df['score'].round(4)
sub_df = sub_df[['ID', 'compatibility_score']]

# Validate
assert len(sub_df) == n_test ** 2
assert sub_df['compatibility_score'].between(0, 1).all()

print(f"  ✓ Shape: {sub_df.shape}")
print(f"  ✓ Range: [{sub_df['compatibility_score'].min():.4f}, {sub_df['compatibility_score'].max():.4f}]")

self_check = sub_df[sub_df['ID'].apply(lambda x: x.split('_')[0] == x.split('_')[1])]
print(f"  ✓ Self-pairs: {len(self_check)}, score={self_check['compatibility_score'].iloc[0]}")

print("\n  Sample:")
print(sub_df.head(10).to_string(index=False))

sub_df.to_csv('submission.csv', index=False)
print("\n  ✓ Saved: submission.csv")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("🎯 ULTIMATE SOLUTION v5.0 COMPLETE! 🎯")
print("=" * 70)
print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║  ULTIMATE v5.0 - FORMULA HUNTER CONFIGURATION                            ║
╠══════════════════════════════════════════════════════════════════════════╣
║  FORMULA DISCOVERY RESULTS:                                              ║
║    • Best single formula: {best_base_col:<20} MSE={formula_results[0][1]:.10f}  ║
║    • Weighted Jaccard:    {'YES' if best_weights else 'NO':<20} MSE={best_weighted_mse:.10f}  ║
║    • Isotonic (CV):       MSE={iso_cv_mse:.10f}                          ║
║    • Polynomial (deg {best_poly_degree}):  MSE={best_poly_mse:.10f}                          ║
║    • Ridge multi-feature: MSE={ridge_cv_mse:.10f}                          ║
║    • Residual-corrected:  MSE={residual_mse:.10f}                          ║
║    • Final blended:       MSE={final_train_mse:.10f}                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║  WINNING CONFIGURATION:                                                  ║
║    • Base Formula: {'Weighted Jaccard' if best_weights else best_base_col:<30}             ║
║    • Transformation: {'ISOTONIC' if iso_cv_mse < best_poly_mse else f'POLYNOMIAL deg {best_poly_degree}':<30}             ║
║    • Alpha: {best_alpha:.4f}                                                     ║
║    • Safety Clamp: {'ENABLED' if USE_SAFETY_CLAMP else 'DISABLED':<30}             ║
╚══════════════════════════════════════════════════════════════════════════╝

OUTPUT: submission.csv ({len(sub_df):,} rows)

🏆 This systematically finds the generator formula for minimum MSE! 🏆
""")
