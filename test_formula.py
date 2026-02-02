import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_csv('/Users/likhith./Desktop/enigma/train.csv')
target_df = pd.read_csv('/Users/likhith./Desktop/enigma/target.csv')

print(f'Train shape: {train_df.shape}')
print(f'Target shape: {target_df.shape}')
print(f'Train columns: {train_df.columns.tolist()}')

# Preprocess
train_df['Business_Interests_Set'] = train_df['Business_Interests'].apply(
    lambda x: set(str(x).split(';')) if pd.notna(x) and x != '' else set()
)
train_df['Business_Objectives_Set'] = train_df['Business_Objectives'].apply(
    lambda x: set(str(x).split(';')) if pd.notna(x) and x != '' else set()
)
train_df['Constraints_Set'] = train_df['Constraints'].apply(
    lambda x: set(str(x).split(';')) if pd.notna(x) and x != '' else set()
)

def jaccard_similarity(set1, set2):
    if len(set1) == 0 and len(set2) == 0:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    if union == 0:
        return 0.0
    return intersection / union

# Test different formulas on a sample
profile_dict = train_df.set_index('Profile_ID').to_dict('index')

sample_pairs = target_df.sample(2000, random_state=42)

concat_preds = []
union_preds = []
interests_preds = []
actuals = []

for _, row in sample_pairs.iterrows():
    src = profile_dict[row['src_user_id']]
    dst = profile_dict[row['dst_user_id']]
    actual = row['compatibility_score']
    
    # Concat Jaccard (prefix each item with field name)
    src_items = set()
    dst_items = set()
    for item in src['Business_Interests_Set']:
        if item: src_items.add(f'BI:{item}')
    for item in src['Business_Objectives_Set']:
        if item: src_items.add(f'BO:{item}')
    for item in src['Constraints_Set']:
        if item: src_items.add(f'CO:{item}')
    for item in dst['Business_Interests_Set']:
        if item: dst_items.add(f'BI:{item}')
    for item in dst['Business_Objectives_Set']:
        if item: dst_items.add(f'BO:{item}')
    for item in dst['Constraints_Set']:
        if item: dst_items.add(f'CO:{item}')
    concat_pred = jaccard_similarity(src_items, dst_items)
    
    # Union Jaccard (all items pooled without prefix)
    src_all = src['Business_Interests_Set'] | src['Business_Objectives_Set'] | src['Constraints_Set']
    dst_all = dst['Business_Interests_Set'] | dst['Business_Objectives_Set'] | dst['Constraints_Set']
    union_pred = jaccard_similarity(src_all, dst_all)
    
    # Interests only
    int_pred = jaccard_similarity(src['Business_Interests_Set'], dst['Business_Interests_Set'])
    
    concat_preds.append(concat_pred)
    union_preds.append(union_pred)
    interests_preds.append(int_pred)
    actuals.append(actual)

print(f'\nMSE Results on 2000 samples:')
print(f'  Concat Jaccard (prefixed): {mean_squared_error(actuals, concat_preds):.8f}')
print(f'  Union Jaccard (pooled): {mean_squared_error(actuals, union_preds):.8f}')
print(f'  Interests Only: {mean_squared_error(actuals, interests_preds):.8f}')

# Check exact matches
concat_exact = sum(1 for a, p in zip(actuals, concat_preds) if abs(a - round(p, 4)) < 0.0001)
union_exact = sum(1 for a, p in zip(actuals, union_preds) if abs(a - round(p, 4)) < 0.0001)
print(f'\nExact matches (within 0.0001):')
print(f'  Concat: {concat_exact}/2000 ({100*concat_exact/2000:.1f}%)')
print(f'  Union: {union_exact}/2000 ({100*union_exact/2000:.1f}%)')

# Check a few specific examples
print("\n\nDetailed examples:")
for i in range(5):
    row = sample_pairs.iloc[i]
    src = profile_dict[row['src_user_id']]
    dst = profile_dict[row['dst_user_id']]
    actual = row['compatibility_score']
    
    src_all = src['Business_Interests_Set'] | src['Business_Objectives_Set'] | src['Constraints_Set']
    dst_all = dst['Business_Interests_Set'] | dst['Business_Objectives_Set'] | dst['Constraints_Set']
    
    inter = src_all & dst_all
    uni = src_all | dst_all
    
    print(f"\nPair {row['src_user_id']}-{row['dst_user_id']}:")
    print(f"  Actual: {actual}")
    print(f"  Union Jaccard: {len(inter)}/{len(uni)} = {len(inter)/len(uni) if len(uni) > 0 else 0:.4f}")
    print(f"  Src items: {src_all}")
    print(f"  Dst items: {dst_all}")
    print(f"  Intersection: {inter}")
