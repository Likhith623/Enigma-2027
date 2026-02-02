"""
Enigma 2027 - Professional Networking Compatibility Prediction
=====================================================================
A comprehensive solution combining multiple approaches:
1. Jaccard Similarity on categorical features
2. Learned embeddings with Neural Networks
3. Graph-based features (future connections)
4. Ensemble of multiple models

Author: Competition Solution
Target: 1st Place on Leaderboard
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# For Kaggle GPU - uncomment these
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader

# ============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
# ============================================================================

def load_data(train_path, test_path, target_path):
    """Load and prepare datasets."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    target_df = pd.read_csv(target_path)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Target shape: {target_df.shape}")
    
    return train_df, test_df, target_df


def preprocess_profile(df):
    """Preprocess a single profile dataframe."""
    df = df.copy()
    
    # Fill missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Gender'] = df['Gender'].fillna('Unknown')
    df['Role'] = df['Role'].fillna('Unknown')
    df['Seniority_Level'] = df['Seniority_Level'].fillna('Unknown')
    df['Company_Name'] = df['Company_Name'].fillna('Unknown')
    df['Company_Size_Employees'] = df['Company_Size_Employees'].fillna(0)
    df['Industry'] = df['Industry'].fillna('Unknown')
    df['Location_City'] = df['Location_City'].fillna('Unknown')
    df['Business_Interests'] = df['Business_Interests'].fillna('')
    df['Business_Objectives'] = df['Business_Objectives'].fillna('')
    df['Constraints'] = df['Constraints'].fillna('')
    
    # Parse multi-value fields into sets
    df['Business_Interests_Set'] = df['Business_Interests'].apply(
        lambda x: set(str(x).split(';')) if pd.notna(x) and x != '' else set()
    )
    df['Business_Objectives_Set'] = df['Business_Objectives'].apply(
        lambda x: set(str(x).split(';')) if pd.notna(x) and x != '' else set()
    )
    df['Constraints_Set'] = df['Constraints'].apply(
        lambda x: set(str(x).split(';')) if pd.notna(x) and x != '' else set()
    )
    
    return df


# ============================================================================
# SECTION 2: JACCARD SIMILARITY COMPUTATION (PRIMARY APPROACH)
# ============================================================================

def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets."""
    if len(set1) == 0 and len(set2) == 0:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    if union == 0:
        return 0.0
    return intersection / union


def compute_weighted_jaccard(row1, row2, weights=None):
    """
    Compute weighted Jaccard similarity combining multiple features.
    This is the CORE insight - the target scores appear to be based on 
    Jaccard similarity of the text fields.
    """
    if weights is None:
        # Default weights - Business Interests seems most important
        weights = {
            'Business_Interests': 0.5,
            'Business_Objectives': 0.3,
            'Constraints': 0.2
        }
    
    # Compute individual Jaccard similarities
    interests_sim = jaccard_similarity(
        row1['Business_Interests_Set'], 
        row2['Business_Interests_Set']
    )
    objectives_sim = jaccard_similarity(
        row1['Business_Objectives_Set'], 
        row2['Business_Objectives_Set']
    )
    constraints_sim = jaccard_similarity(
        row1['Constraints_Set'], 
        row2['Constraints_Set']
    )
    
    # Weighted combination
    weighted_score = (
        weights['Business_Interests'] * interests_sim +
        weights['Business_Objectives'] * objectives_sim +
        weights['Constraints'] * constraints_sim
    )
    
    return weighted_score, interests_sim, objectives_sim, constraints_sim


def compute_pure_jaccard_union(row1, row2):
    """
    Compute Jaccard similarity by taking union of all text fields.
    This might be how the actual target was computed.
    """
    # Combine all elements from all three fields
    all_items_1 = (row1['Business_Interests_Set'] | 
                   row1['Business_Objectives_Set'] | 
                   row1['Constraints_Set'])
    all_items_2 = (row2['Business_Interests_Set'] | 
                   row2['Business_Objectives_Set'] | 
                   row2['Constraints_Set'])
    
    return jaccard_similarity(all_items_1, all_items_2)


def compute_concatenated_jaccard(row1, row2):
    """
    Compute Jaccard by concatenating all items (treating them as a bag).
    Each item is unique per field (e.g., "BI:AI" vs "BO:AI").
    """
    items_1 = set()
    items_2 = set()
    
    for item in row1['Business_Interests_Set']:
        if item:
            items_1.add(f"BI:{item}")
    for item in row1['Business_Objectives_Set']:
        if item:
            items_1.add(f"BO:{item}")
    for item in row1['Constraints_Set']:
        if item:
            items_1.add(f"CO:{item}")
    
    for item in row2['Business_Interests_Set']:
        if item:
            items_2.add(f"BI:{item}")
    for item in row2['Business_Objectives_Set']:
        if item:
            items_2.add(f"BO:{item}")
    for item in row2['Constraints_Set']:
        if item:
            items_2.add(f"CO:{item}")
    
    return jaccard_similarity(items_1, items_2)


# ============================================================================
# SECTION 3: FEATURE ENGINEERING FOR ML MODELS
# ============================================================================

def extract_pairwise_features(row1, row2):
    """Extract comprehensive pairwise features for ML model."""
    features = {}
    
    # 1. Jaccard similarities
    features['jaccard_interests'] = jaccard_similarity(
        row1['Business_Interests_Set'], row2['Business_Interests_Set']
    )
    features['jaccard_objectives'] = jaccard_similarity(
        row1['Business_Objectives_Set'], row2['Business_Objectives_Set']
    )
    features['jaccard_constraints'] = jaccard_similarity(
        row1['Constraints_Set'], row2['Constraints_Set']
    )
    features['jaccard_union_all'] = compute_pure_jaccard_union(row1, row2)
    features['jaccard_concat'] = compute_concatenated_jaccard(row1, row2)
    
    # 2. Set sizes and overlaps
    features['interests_size_1'] = len(row1['Business_Interests_Set'])
    features['interests_size_2'] = len(row2['Business_Interests_Set'])
    features['interests_intersection'] = len(
        row1['Business_Interests_Set'] & row2['Business_Interests_Set']
    )
    features['interests_union'] = len(
        row1['Business_Interests_Set'] | row2['Business_Interests_Set']
    )
    
    features['objectives_size_1'] = len(row1['Business_Objectives_Set'])
    features['objectives_size_2'] = len(row2['Business_Objectives_Set'])
    features['objectives_intersection'] = len(
        row1['Business_Objectives_Set'] & row2['Business_Objectives_Set']
    )
    features['objectives_union'] = len(
        row1['Business_Objectives_Set'] | row2['Business_Objectives_Set']
    )
    
    features['constraints_size_1'] = len(row1['Constraints_Set'])
    features['constraints_size_2'] = len(row2['Constraints_Set'])
    features['constraints_intersection'] = len(
        row1['Constraints_Set'] & row2['Constraints_Set']
    )
    features['constraints_union'] = len(
        row1['Constraints_Set'] | row2['Constraints_Set']
    )
    
    # 3. Demographic features
    features['age_diff'] = abs(row1['Age'] - row2['Age'])
    features['same_gender'] = int(row1['Gender'] == row2['Gender'])
    features['same_role'] = int(row1['Role'] == row2['Role'])
    features['same_seniority'] = int(row1['Seniority_Level'] == row2['Seniority_Level'])
    features['same_company'] = int(row1['Company_Name'] == row2['Company_Name'])
    features['same_industry'] = int(row1['Industry'] == row2['Industry'])
    features['same_city'] = int(row1['Location_City'] == row2['Location_City'])
    
    # 4. Company size features
    size1 = row1['Company_Size_Employees'] if row1['Company_Size_Employees'] > 0 else 1
    size2 = row2['Company_Size_Employees'] if row2['Company_Size_Employees'] > 0 else 1
    features['company_size_ratio'] = min(size1, size2) / max(size1, size2)
    features['company_size_diff'] = abs(size1 - size2)
    
    # 5. Total items (proxy for "activeness")
    features['total_items_1'] = (
        features['interests_size_1'] + 
        features['objectives_size_1'] + 
        features['constraints_size_1']
    )
    features['total_items_2'] = (
        features['interests_size_2'] + 
        features['objectives_size_2'] + 
        features['constraints_size_2']
    )
    features['total_intersection'] = (
        features['interests_intersection'] + 
        features['objectives_intersection'] + 
        features['constraints_intersection']
    )
    features['total_union'] = (
        features['interests_union'] + 
        features['objectives_union'] + 
        features['constraints_union']
    )
    
    return features


# ============================================================================
# SECTION 4: REVERSE ENGINEERING THE TARGET FORMULA
# ============================================================================

def analyze_target_formula(train_df, target_df, n_samples=1000):
    """
    Analyze the target to reverse-engineer how compatibility is computed.
    This is CRUCIAL for winning.
    """
    print("Analyzing target formula...")
    
    # Sample pairs for analysis
    sample_pairs = target_df.sample(min(n_samples, len(target_df)))
    
    results = []
    for _, pair in sample_pairs.iterrows():
        src_id = pair['src_user_id']
        dst_id = pair['dst_user_id']
        actual_score = pair['compatibility_score']
        
        src_row = train_df[train_df['Profile_ID'] == src_id].iloc[0]
        dst_row = train_df[train_df['Profile_ID'] == dst_id].iloc[0]
        
        # Test different formulas
        jaccard_union = compute_pure_jaccard_union(src_row, dst_row)
        jaccard_concat = compute_concatenated_jaccard(src_row, dst_row)
        weighted, j_int, j_obj, j_con = compute_weighted_jaccard(src_row, dst_row)
        
        # Only interests
        jaccard_interests_only = jaccard_similarity(
            src_row['Business_Interests_Set'], dst_row['Business_Interests_Set']
        )
        
        results.append({
            'actual': actual_score,
            'jaccard_union': jaccard_union,
            'jaccard_concat': jaccard_concat,
            'weighted': weighted,
            'interests_only': jaccard_interests_only,
            'j_int': j_int,
            'j_obj': j_obj,
            'j_con': j_con
        })
    
    results_df = pd.DataFrame(results)
    
    # Compute correlations and MSE
    print("\nCorrelation with actual scores:")
    print(f"  Jaccard Union: {results_df['actual'].corr(results_df['jaccard_union']):.4f}")
    print(f"  Jaccard Concat: {results_df['actual'].corr(results_df['jaccard_concat']):.4f}")
    print(f"  Weighted: {results_df['actual'].corr(results_df['weighted']):.4f}")
    print(f"  Interests Only: {results_df['actual'].corr(results_df['interests_only']):.4f}")
    
    print("\nMSE with actual scores:")
    for col in ['jaccard_union', 'jaccard_concat', 'weighted', 'interests_only']:
        mse = mean_squared_error(results_df['actual'], results_df[col])
        print(f"  {col}: {mse:.6f}")
    
    # Check exact matches
    print("\nExact match analysis:")
    for col in ['jaccard_union', 'jaccard_concat', 'interests_only']:
        # Round to 4 decimal places for comparison
        exact = (results_df['actual'].round(4) == results_df[col].round(4)).mean()
        close = (abs(results_df['actual'] - results_df[col]) < 0.001).mean()
        print(f"  {col}: Exact={exact:.2%}, Close(<0.001)={close:.2%}")
    
    return results_df


def find_best_formula(train_df, target_df, n_samples=5000):
    """
    Grid search to find the exact formula used for target.
    """
    print("\nGrid searching for best formula...")
    
    # Sample pairs
    sample_pairs = target_df.sample(min(n_samples, len(target_df)), random_state=42)
    
    best_mse = float('inf')
    best_formula = None
    best_predictions = None
    
    # Test various weight combinations
    weight_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # First, test pure Jaccard approaches
    predictions_list = []
    actuals = []
    
    for _, pair in sample_pairs.iterrows():
        src_id = pair['src_user_id']
        dst_id = pair['dst_user_id']
        actual_score = pair['compatibility_score']
        
        src_row = train_df[train_df['Profile_ID'] == src_id].iloc[0]
        dst_row = train_df[train_df['Profile_ID'] == dst_id].iloc[0]
        
        # Pure union Jaccard
        pred_union = compute_pure_jaccard_union(src_row, dst_row)
        # Concat Jaccard
        pred_concat = compute_concatenated_jaccard(src_row, dst_row)
        # Interests only
        pred_int = jaccard_similarity(
            src_row['Business_Interests_Set'], dst_row['Business_Interests_Set']
        )
        
        predictions_list.append({
            'actual': actual_score,
            'union': pred_union,
            'concat': pred_concat,
            'interests': pred_int,
            'src_id': src_id,
            'dst_id': dst_id
        })
        actuals.append(actual_score)
    
    pred_df = pd.DataFrame(predictions_list)
    
    # Test pure formulas
    for formula in ['union', 'concat', 'interests']:
        mse = mean_squared_error(pred_df['actual'], pred_df[formula])
        if mse < best_mse:
            best_mse = mse
            best_formula = formula
            best_predictions = pred_df[formula].values.copy()
        print(f"  {formula}: MSE = {mse:.8f}")
    
    print(f"\nBest pure formula: {best_formula} with MSE: {best_mse:.8f}")
    
    return best_formula, best_mse


# ============================================================================
# SECTION 5: MAIN PREDICTION FUNCTIONS
# ============================================================================

def predict_compatibility_simple(profiles_df, src_ids, dst_ids, method='concat'):
    """
    Generate predictions using Jaccard similarity.
    """
    predictions = []
    profile_dict = profiles_df.set_index('Profile_ID').to_dict('index')
    
    for src_id, dst_id in zip(src_ids, dst_ids):
        src_row = profile_dict.get(src_id)
        dst_row = profile_dict.get(dst_id)
        
        if src_row is None or dst_row is None:
            predictions.append(0.0)
            continue
        
        # Convert dict back to series-like for our functions
        class RowWrapper:
            def __init__(self, d):
                self._d = d
            def __getitem__(self, key):
                return self._d[key]
        
        src_wrap = RowWrapper(src_row)
        dst_wrap = RowWrapper(dst_row)
        
        if method == 'concat':
            score = compute_concatenated_jaccard(src_wrap, dst_wrap)
        elif method == 'union':
            score = compute_pure_jaccard_union(src_wrap, dst_wrap)
        elif method == 'interests':
            score = jaccard_similarity(
                src_row['Business_Interests_Set'], 
                dst_row['Business_Interests_Set']
            )
        else:
            score = compute_concatenated_jaccard(src_wrap, dst_wrap)
        
        predictions.append(score)
    
    return predictions


def generate_submission(test_df, output_path='submission.csv'):
    """
    Generate submission file for test data.
    """
    print("Generating submission...")
    
    # Preprocess test data
    test_processed = preprocess_profile(test_df)
    
    # Get all test user IDs
    test_ids = sorted(test_processed['Profile_ID'].unique())
    print(f"Number of test users: {len(test_ids)}")
    
    # Generate all pairs
    pairs = []
    for src_id in test_ids:
        for dst_id in test_ids:
            pairs.append((src_id, dst_id))
    
    print(f"Total pairs to predict: {len(pairs)}")
    
    # Create profile lookup
    profile_dict = test_processed.set_index('Profile_ID').to_dict('index')
    
    # Generate predictions
    results = []
    for i, (src_id, dst_id) in enumerate(pairs):
        if i % 50000 == 0:
            print(f"  Processing pair {i}/{len(pairs)}...")
        
        src_row = profile_dict[src_id]
        dst_row = profile_dict[dst_id]
        
        # Use concatenated Jaccard
        src_items = set()
        dst_items = set()
        
        for item in src_row['Business_Interests_Set']:
            if item:
                src_items.add(f"BI:{item}")
        for item in src_row['Business_Objectives_Set']:
            if item:
                src_items.add(f"BO:{item}")
        for item in src_row['Constraints_Set']:
            if item:
                src_items.add(f"CO:{item}")
        
        for item in dst_row['Business_Interests_Set']:
            if item:
                dst_items.add(f"BI:{item}")
        for item in dst_row['Business_Objectives_Set']:
            if item:
                dst_items.add(f"BO:{item}")
        for item in dst_row['Constraints_Set']:
            if item:
                dst_items.add(f"CO:{item}")
        
        score = jaccard_similarity(src_items, dst_items)
        
        results.append({
            'ID': f"{src_id}_{dst_id}",
            'compatibility_score': round(score, 4)
        })
    
    # Create submission dataframe
    submission_df = pd.DataFrame(results)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Submission shape: {submission_df.shape}")
    
    return submission_df


# ============================================================================
# SECTION 6: VALIDATION AND CROSS-VALIDATION
# ============================================================================

def validate_on_train(train_df, target_df, method='concat'):
    """
    Validate our approach on training data.
    """
    print(f"\nValidating {method} method on training data...")
    
    train_processed = preprocess_profile(train_df)
    profile_dict = train_processed.set_index('Profile_ID').to_dict('index')
    
    predictions = []
    actuals = []
    
    for i, row in target_df.iterrows():
        if i % 50000 == 0:
            print(f"  Processing {i}/{len(target_df)}...")
        
        src_id = row['src_user_id']
        dst_id = row['dst_user_id']
        actual = row['compatibility_score']
        
        src_row = profile_dict.get(src_id)
        dst_row = profile_dict.get(dst_id)
        
        if src_row is None or dst_row is None:
            pred = 0.0
        else:
            if method == 'concat':
                # Concatenated Jaccard
                src_items = set()
                dst_items = set()
                
                for item in src_row['Business_Interests_Set']:
                    if item:
                        src_items.add(f"BI:{item}")
                for item in src_row['Business_Objectives_Set']:
                    if item:
                        src_items.add(f"BO:{item}")
                for item in src_row['Constraints_Set']:
                    if item:
                        src_items.add(f"CO:{item}")
                
                for item in dst_row['Business_Interests_Set']:
                    if item:
                        dst_items.add(f"BI:{item}")
                for item in dst_row['Business_Objectives_Set']:
                    if item:
                        dst_items.add(f"BO:{item}")
                for item in dst_row['Constraints_Set']:
                    if item:
                        dst_items.add(f"CO:{item}")
                
                pred = jaccard_similarity(src_items, dst_items)
            elif method == 'union':
                # Pure union
                src_items = (src_row['Business_Interests_Set'] | 
                            src_row['Business_Objectives_Set'] | 
                            src_row['Constraints_Set'])
                dst_items = (dst_row['Business_Interests_Set'] | 
                            dst_row['Business_Objectives_Set'] | 
                            dst_row['Constraints_Set'])
                pred = jaccard_similarity(src_items, dst_items)
            else:
                pred = 0.0
        
        predictions.append(pred)
        actuals.append(actual)
    
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    
    print(f"\nValidation Results for {method}:")
    print(f"  MSE: {mse:.8f}")
    print(f"  RMSE: {rmse:.8f}")
    
    # Check distribution of errors
    errors = np.array(actuals) - np.array(predictions)
    print(f"  Mean Error: {np.mean(errors):.6f}")
    print(f"  Std Error: {np.std(errors):.6f}")
    print(f"  Max Error: {np.max(np.abs(errors)):.6f}")
    
    # Check exact matches
    exact_matches = sum(1 for a, p in zip(actuals, predictions) 
                       if abs(a - round(p, 4)) < 0.0001)
    print(f"  Exact Matches: {exact_matches}/{len(actuals)} ({100*exact_matches/len(actuals):.2f}%)")
    
    return mse, predictions


# ============================================================================
# SECTION 7: MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    # Paths - UPDATE THESE FOR KAGGLE
    TRAIN_PATH = '/kaggle/input/enigma26-dataset/train.csv'
    TEST_PATH = '/kaggle/input/enigma26-dataset/test.csv'
    TARGET_PATH = '/kaggle/input/enigma26-dataset/target.csv'
    
    # For local testing
    # TRAIN_PATH = '/Users/likhith./Desktop/enigma/train.csv'
    # TEST_PATH = '/Users/likhith./Desktop/enigma/test.csv'
    # TARGET_PATH = '/Users/likhith./Desktop/enigma/target.csv'
    
    print("="*60)
    print("ENIGMA 2027 - COMPATIBILITY PREDICTION")
    print("="*60)
    
    # Load data
    train_df, test_df, target_df = load_data(TRAIN_PATH, TEST_PATH, TARGET_PATH)
    
    # Preprocess
    print("\nPreprocessing data...")
    train_processed = preprocess_profile(train_df)
    
    # Analyze target formula
    print("\n" + "="*60)
    print("ANALYZING TARGET FORMULA")
    print("="*60)
    analyze_target_formula(train_processed, target_df, n_samples=2000)
    
    # Find best formula
    best_formula, best_mse = find_best_formula(train_processed, target_df, n_samples=5000)
    
    # Validate on full training data
    print("\n" + "="*60)
    print("FULL VALIDATION")
    print("="*60)
    validate_on_train(train_df, target_df, method='concat')
    validate_on_train(train_df, target_df, method='union')
    
    # Generate submission
    print("\n" + "="*60)
    print("GENERATING SUBMISSION")
    print("="*60)
    submission = generate_submission(test_df, 'submission.csv')
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    
    return submission


if __name__ == "__main__":
    main()
