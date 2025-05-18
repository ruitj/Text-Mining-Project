
import os
import numpy as np
import pandas as pd
import json
from scipy import sparse

def load_data(preprocessing_version, feature_type, feature_engineering_dir="../feature_engineered_data", processed_data_dir="../processed_data"):
    """
    Load data from both processed_data and feature_engineered_data directories.
    
    Args:
        preprocessing_version: Name of preprocessing version (e.g., 'regexp_snowball')
        feature_type: Type of features to use ('tfidf', 'word2vec', 'mini_sbert')
        feature_engineering_dir: Directory with feature engineered data
        processed_data_dir: Directory with processed text data
        
    Returns:
        Dictionary with loaded data and features ready for training
    """
    print(f"Loading data for {preprocessing_version} with {feature_type} features")
    
    # Paths for processed text data
    proc_version_dir = os.path.join(processed_data_dir, preprocessing_version)
    train_file = os.path.join(proc_version_dir, 'train.csv')
    val_file = os.path.join(proc_version_dir, 'val.csv')
    test_file = os.path.join(proc_version_dir, 'test.csv')
    
    # Check if processed text files exist
    if not (os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file)):
        raise FileNotFoundError(f"Missing processed data files for {preprocessing_version}")
    
    # Load processed text data
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)
    
    print(f"Loaded processed text data - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Path for feature engineered data
    feat_version_dir = os.path.join(feature_engineering_dir, preprocessing_version)
    
    # Load feature info to check available features
    try:
        with open(os.path.join(feat_version_dir, 'feature_info.json'), 'r') as f:
            feature_info = json.load(f)
        print(f"Feature info: {feature_info}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing feature info for {preprocessing_version}")
    
    # Check if requested feature type is available
    if f"{feature_type}_shape" not in feature_info:
        raise ValueError(f"Feature type '{feature_type}' not available for {preprocessing_version}")
    
    # Load feature tensors - using a simple approach with numpy files
    try:
        # Determine appropriate loading method based on feature type
        if feature_type == 'tfidf':
            # For sparse matrices like TF-IDF
            X_train = sparse.load_npz(os.path.join(feat_version_dir, f'{feature_type}_X_train.npz'))
            X_val = sparse.load_npz(os.path.join(feat_version_dir, f'{feature_type}_X_val.npz'))
            X_test = sparse.load_npz(os.path.join(feat_version_dir, f'{feature_type}_X_test.npz'))
        else:
            # For dense arrays like word2vec or mini_sbert
            X_train = np.load(os.path.join(feat_version_dir, f'{feature_type}_X_train.npy'))
            X_val = np.load(os.path.join(feat_version_dir, f'{feature_type}_X_val.npy'))
            X_test = np.load(os.path.join(feat_version_dir, f'{feature_type}_X_test.npy'))
        
        print(f"Loaded {feature_type} features - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    except FileNotFoundError:
        # If feature files aren't available, provide a helpful error
        raise FileNotFoundError(
            f"Feature files for {feature_type} not found in {feat_version_dir}. "
            f"Make sure you have run the feature engineering process for this combination."
        )
    
    # Get labels
    y_train = train_df['label'].values if 'label' in train_df.columns else None
    y_val = val_df['label'].values if 'label' in val_df.columns else None
    
    # Get class weights if available
    if 'class_weights' in feature_info:
        class_weights = feature_info['class_weights']
        # Convert string representation back to dictionary if needed
        if isinstance(class_weights, str):
            import ast
            try:
                class_weights = ast.literal_eval(class_weights)
            except:
                class_weights = None
    else:
        class_weights = None
    
    # Return everything we need for model training
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'class_weights': class_weights,
        'feature_type': feature_type,
        'preprocessing_version': preprocessing_version
    }

def get_all_combinations():
    """
    Helper function to get all valid preprocessing and feature type combinations.
    
    Returns:
        List of tuples (preprocessing_version, feature_type)
    """
    # Default preprocessing versions
    preprocessing_versions = ['regexp_snowball', 'tweet_base', 'tweet_porter', 
                              'whitespace_lancaster', 'word_lemma']
    
    # Default feature types
    feature_types = ['tfidf', 'word2vec', 'mini_sbert']
    
    # Return all combinations
    return [(p, f) for p in preprocessing_versions for f in feature_types]

def save_results(model_results, model_name, preprocessing_version, feature_type, results_dir="model_results"):
    """
    Save model results to a standardized directory structure.
    
    Args:
        model_results: Dictionary with model results
        model_name: Name of the model (e.g., 'knn', 'logreg')
        preprocessing_version: Name of preprocessing version used
        feature_type: Type of features used
        results_dir: Base directory to save results
    """
    # Create output directory
    model_dir = os.path.join(results_dir, preprocessing_version, feature_type, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save metrics
    metrics = {
        'preprocessing_version': preprocessing_version,
        'feature_type': feature_type,
        'model_name': model_name,
        'train_accuracy': model_results.get('train_accuracy', None),
        'train_f1': model_results.get('train_f1', None),
        'val_accuracy': model_results.get('val_accuracy', None),
        'val_f1': model_results.get('val_f1', None),
        'parameters': model_results.get('params', {})
    }
    
    with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save predictions if available
    if 'predictions' in model_results and 'test' in model_results['predictions']:
        predictions_df = pd.DataFrame({
            'id': range(len(model_results['predictions']['test'])),
            'prediction': model_results['predictions']['test']
        })
        predictions_df.to_csv(os.path.join(model_dir, 'test_predictions.csv'), index=False)
    
    print(f"Results saved to: {model_dir}")
    
    return model_dir