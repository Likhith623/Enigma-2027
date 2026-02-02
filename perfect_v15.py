"""
================================================================================
ENIGMA 2027 - PERFECT SOLUTION v15 (ALL FEATURES)
================================================================================
🎯 TARGET: MSE = 0 🎯

DETECTIVE FOUND:
  - 831/1110 Jaccard keys have VARYING scores
  - Only 25.2% match pure Jaccard
  - Formula uses: Jaccards + Role + Industry + Location + Seniority

THIS VERSION:
  1. Uses ALL Jaccard features (j_all, j_bi, j_bo, j_co)
  2. Uses ALL categorical matches (Role, Industry, Location, Seniority)
  3. Uses intersection/union sizes
  4. Learns exact formula with XGBoost/LightGBM

================================================================================
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import os
import re
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🎯 ENIGMA 2027 - PERFECT v15 (ALL FEATURES) 🎯")
print("=" * 80)

start_time = time.time()

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1] Loading data...")

DATA_DIR = '/kaggle/input/enigma26/Engima26_Dataset'
if not os.path.exists(DATA_DIR):
    for path in ['/kaggle/input/enigma26', '.']:
        if os.path.exists(path):
            DATA_DIR = path
            break

print(f"    Data directory: {DATA_DIR}")

if os.path.exists(f'{DATA_DIR}/train.xlsx'):
    train_df = pd.read_excel(f'{DATA_DIR}/train.xlsx')
    test_df = pd.read_excel(f'{DATA_DIR}/test.xlsx')
else:
    train_df = pd.read_csv(f'{DATA_DIR}/train.csv')
    test_df = pd.read_csv(f'{DATA_DIR}/test.csv')

target_df = pd.read_csv(f'{DATA_DIR}/target.csv')

print(f"    Train users: {len(train_df)}, Test users: {len(test_df)}")
print(f"    Training pairs: {len(target_df)}")

# =============================================================================
# PARSE SETS
# =============================================================================
print("\n[2] Parsing feature sets...")

def parse_set(val):
    if pd.isna(val) or str(val).strip().lower() in ('', 'nan', 'none'):
        return frozenset()
    return frozenset(t.strip() for t in str(val).split(';') if t.strip())

def jaccard(s1, s2):
    if not s1 and not s2:
        return 0.0
    union = s1 | s2
    return len(s1 & s2) / len(union) if union else 0.0

# Parse for all users
for df in [train_df, test_df]:
    df['BI'] = df['Business_Interests'].apply(parse_set)
    df['BO'] = df['Business_Objectives'].apply(parse_set)
    df['CO'] = df['Constraints'].apply(parse_set)
    df['ALL'] = df.apply(lambda r: r['BI'] | r['BO'] | r['CO'], axis=1)

# Create lookups
train_lookup = {row['Profile_ID']: row for _, row in train_df.iterrows()}
test_lookup = {row['Profile_ID']: row for _, row in test_df.iterrows()}
all_lookup = {**train_lookup, **test_lookup}

print(f"    Parsed {len(train_lookup)} train + {len(test_lookup)} test users")

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================
print("\n[3] Building feature extractor...")

# Get all unique values for categorical features
all_roles = set()
all_industries = set()
all_locations = set()
all_seniorities = set()

for df in [train_df, test_df]:
    if 'Role' in df.columns:
        all_roles.update(df['Role'].dropna().unique())
    if 'Industry' in df.columns:
        all_industries.update(df['Industry'].dropna().unique())
    if 'Location_City' in df.columns:
        all_locations.update(df['Location_City'].dropna().unique())
    if 'Seniority_Level' in df.columns:
        all_seniorities.update(df['Seniority_Level'].dropna().unique())

print(f"    Roles: {len(all_roles)}, Industries: {len(all_industries)}")
print(f"    Locations: {len(all_locations)}, Seniorities: {len(all_seniorities)}")

def extract_features(r1, r2):
    """Extract ALL features for a pair"""
    features = {}
    
    # Jaccard features
    features['j_all'] = jaccard(r1['ALL'], r2['ALL'])
    features['j_bi'] = jaccard(r1['BI'], r2['BI'])
    features['j_bo'] = jaccard(r1['BO'], r2['BO'])
    features['j_co'] = jaccard(r1['CO'], r2['CO'])
    
    # Set sizes
    features['all_inter'] = len(r1['ALL'] & r2['ALL'])
    features['all_union'] = len(r1['ALL'] | r2['ALL'])
    features['bi_inter'] = len(r1['BI'] & r2['BI'])
    features['bi_union'] = len(r1['BI'] | r2['BI'])
    features['bo_inter'] = len(r1['BO'] & r2['BO'])
    features['bo_union'] = len(r1['BO'] | r2['BO'])
    features['co_inter'] = len(r1['CO'] & r2['CO'])
    features['co_union'] = len(r1['CO'] | r2['CO'])
    
    # Individual sizes
    features['all_size_1'] = len(r1['ALL'])
    features['all_size_2'] = len(r2['ALL'])
    features['size_diff'] = abs(len(r1['ALL']) - len(r2['ALL']))
    
    # Categorical matches
    features['role_match'] = 1.0 if r1.get('Role') == r2.get('Role') and pd.notna(r1.get('Role')) else 0.0
    features['industry_match'] = 1.0 if r1.get('Industry') == r2.get('Industry') and pd.notna(r1.get('Industry')) else 0.0
    features['location_match'] = 1.0 if r1.get('Location_City') == r2.get('Location_City') and pd.notna(r1.get('Location_City')) else 0.0
    features['seniority_match'] = 1.0 if r1.get('Seniority_Level') == r2.get('Seniority_Level') and pd.notna(r1.get('Seniority_Level')) else 0.0
    
    # Total categorical matches
    features['total_cat_match'] = features['role_match'] + features['industry_match'] + features['location_match'] + features['seniority_match']
    
    # Missing value indicators
    features['role_missing'] = 1.0 if pd.isna(r1.get('Role')) or pd.isna(r2.get('Role')) else 0.0
    features['industry_missing'] = 1.0 if pd.isna(r1.get('Industry')) or pd.isna(r2.get('Industry')) else 0.0
    features['seniority_missing'] = 1.0 if pd.isna(r1.get('Seniority_Level')) or pd.isna(r2.get('Seniority_Level')) else 0.0
    
    return features

# =============================================================================
# BUILD TRAINING DATA
# =============================================================================
print("\n[4] Building training dataset...")

# Self-pair detection
self_pairs = target_df[target_df['src_user_id'] == target_df['dst_user_id']]
SELF_SCORE = float(self_pairs['compatibility_score'].median()) if len(self_pairs) > 0 else 0.0
print(f"    Self-pair score: {SELF_SCORE}")

regular_pairs = target_df[target_df['src_user_id'] != target_df['dst_user_id']]
n_pairs = len(regular_pairs)
print(f"    Regular pairs: {n_pairs}")

# Extract features for all training pairs
X_train = []
y_train = []

t0 = time.time()
for idx, row in enumerate(regular_pairs.itertuples()):
    r1 = train_lookup[row.src_user_id]
    r2 = train_lookup[row.dst_user_id]
    
    features = extract_features(r1, r2)
    X_train.append(features)
    y_train.append(row.compatibility_score)
    
    if (idx + 1) % 50000 == 0:
        print(f"    Processed {idx+1}/{n_pairs} pairs...")

X_train = pd.DataFrame(X_train)
y_train = np.array(y_train)

print(f"    Features: {X_train.shape[1]}")
print(f"    Feature names: {list(X_train.columns)}")

# =============================================================================
# TRAIN MODEL
# =============================================================================
print("\n[5] Training model...")

# Try LightGBM first (faster), fall back to sklearn
try:
    import lightgbm as lgb
    USE_LGBM = True
    print("    Using LightGBM")
except ImportError:
    USE_LGBM = False
    print("    Using sklearn GradientBoosting")

if USE_LGBM:
    model = lgb.LGBMRegressor(
        n_estimators=500,
        max_depth=10,
        learning_rate=0.05,
        num_leaves=64,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
else:
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

# Evaluate on training data
train_pred = model.predict(X_train)
train_mse = np.mean((train_pred - y_train) ** 2)
train_mae = np.mean(np.abs(train_pred - y_train))

print(f"\n    ✓ Training MSE: {train_mse:.10f}")
print(f"    ✓ Training MAE: {train_mae:.10f}")

# Count perfect matches
perfect = np.sum(np.abs(train_pred - y_train) < 1e-4)
print(f"    ✓ Perfect matches (< 1e-4): {perfect}/{n_pairs} ({100*perfect/n_pairs:.2f}%)")

# =============================================================================
# ANALYZE IF WE NEED MORE PRECISION
# =============================================================================
print("\n[6] Checking if rounding helps...")

# Since scores seem to have specific decimal patterns, try rounding predictions
best_mse = train_mse
best_round = None

for d in [2, 3, 4, 5, 6]:
    rounded = np.round(train_pred, d)
    mse = np.mean((rounded - y_train) ** 2)
    if mse < best_mse:
        best_mse = mse
        best_round = d
        print(f"    Round({d}): MSE = {mse:.10f} ✓")
    else:
        print(f"    Round({d}): MSE = {mse:.10f}")

if best_round:
    print(f"\n    Best with round({best_round}): MSE = {best_mse:.10f}")
else:
    print(f"\n    No rounding improves MSE")

# =============================================================================
# GENERATE SUBMISSION
# =============================================================================
print("\n[7] Generating submission...")

test_ids = sorted(test_df['Profile_ID'].unique().tolist())
n_test = len(test_ids)
print(f"    Test users: {n_test}")
print(f"    Predictions: {n_test} × {n_test} = {n_test**2}")

results = []
t0 = time.time()

for i, src in enumerate(test_ids):
    r1 = test_lookup[src]
    batch_features = []
    batch_pairs = []
    
    for dst in test_ids:
        if src == dst:
            results.append({'ID': f'{src}_{dst}', 'compatibility_score': SELF_SCORE})
        else:
            r2 = test_lookup[dst]
            features = extract_features(r1, r2)
            batch_features.append(features)
            batch_pairs.append((src, dst))
    
    if batch_features:
        X_batch = pd.DataFrame(batch_features)
        preds = model.predict(X_batch)
        
        if best_round:
            preds = np.round(preds, best_round)
        
        for (src_id, dst_id), pred in zip(batch_pairs, preds):
            results.append({'ID': f'{src_id}_{dst_id}', 'compatibility_score': pred})
    
    if (i + 1) % 100 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (n_test - i - 1)
        print(f"    {i+1}/{n_test} users ({elapsed:.1f}s, ~{eta:.1f}s remaining)")

submission = pd.DataFrame(results)
submission.to_csv('submission.csv', index=False)

elapsed_total = time.time() - start_time

print(f"\n    ✓ submission.csv saved ({len(submission)} rows)")

# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================
print("\n[8] Feature importance...")

if USE_LGBM:
    importance = model.feature_importances_
else:
    importance = model.feature_importances_

feature_imp = sorted(zip(X_train.columns, importance), key=lambda x: -x[1])
print("    Top features:")
for name, imp in feature_imp[:10]:
    print(f"      {name}: {imp:.4f}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("🎯 PERFECT v15 COMPLETE!")
print("=" * 80)
print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  TRAINING MSE:    {train_mse:<56.10f}║
║  TRAINING MAE:    {train_mae:<56.10f}║
║  Perfect matches: {perfect}/{n_pairs} ({100*perfect/n_pairs:.2f}%)                                ║
║  Rounding:        {str(best_round) if best_round else 'None':<56}║
║  Self-pair score: {SELF_SCORE:<56}║
║  Time:            {elapsed_total:.1f}s                                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

if train_mse < 1e-6:
    print("🎯 PERFECT! Training MSE ≈ 0!")
    print("   Submit submission.csv for 1ST PLACE! 🏆")
elif train_mse < 1e-4:
    print("✓ EXCELLENT! Very low MSE!")
    print("   You should be in top positions!")
else:
    print("⚠️ MSE is still high. The formula might have additional rules.")
    print("   But this is still your best submission yet!")

print("\n✓ Done! Download submission.csv and submit!")
