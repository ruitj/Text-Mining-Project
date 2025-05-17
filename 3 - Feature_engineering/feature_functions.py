
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
import json
from collections import Counter
from sklearn.utils import resample  # For undersampling/oversampling
from imblearn.over_sampling import SMOTE  # For SMOTE strategy

warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Then add the FeatureEngineering class...

class FeatureEngineering:
    """
    Feature engineering for stock sentiment analysis,
    with methods to address class imbalance.
    """
    def __init__(self, base_dir="processed_data", output_dir="feature_engineered_data"):
        self.base_dir = base_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        # Get preprocessing versions
        self.preprocessing_versions = [folder for folder in os.listdir(base_dir) 
                                      if os.path.isdir(os.path.join(base_dir, folder))]
        print(f"Found preprocessing versions: {self.preprocessing_versions}")
    
    def load_data(self, preprocessing_version):
        """
        Load preprocessed data for a specific version.
        """
        version_dir = os.path.join(self.base_dir, preprocessing_version)
        
        train_file = os.path.join(version_dir, 'train.csv')
        val_file = os.path.join(version_dir, 'val.csv')
        test_file = os.path.join(version_dir, 'test.csv')
        
        if not (os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file)):
            print(f"Missing data files for {preprocessing_version}")
            return None, None, None
        
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(test_file)
        
        print(f"Loaded {preprocessing_version} data - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Analyze class distribution
        if 'label' in train_df.columns:
            class_counts = train_df['label'].value_counts().sort_index()
            print("Class distribution in training set:")
            for label, count in class_counts.items():
                print(f"  Class {label}: {count} ({count/len(train_df)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def handle_class_imbalance(self, train_df, strategy="weight", random_state=42):
        """
        Handle class imbalance in the training data.
        
        Args:
            train_df: Training DataFrame
            strategy: Strategy to handle imbalance ('weight', 'undersample', 'oversample', 'smote')
            random_state: Random seed for reproducibility
            
        Returns:
            Processed DataFrame and class weights (if applicable)
        """
        if 'label' not in train_df.columns:
            print("No label column found in training data")
            return train_df, None
        
        # Calculate class weights
        class_counts = train_df['label'].value_counts().sort_index()
        n_samples = len(train_df)
        n_classes = len(class_counts)
        
        # Compute weights inversely proportional to class frequencies
        class_weights = {label: n_samples / (n_classes * count) 
                         for label, count in class_counts.items()}
        
        print(f"Using {strategy} strategy to handle class imbalance")
        print("Class weights:", class_weights)
        
        if strategy == "weight":
            # Just return the original data with calculated weights
            return train_df, class_weights
        
        elif strategy == "undersample":
            # Undersample majority classes
            from sklearn.utils import resample
            
            # Find the minority class
            min_class = class_counts.idxmin()
            min_count = class_counts[min_class]
            
            # Undersample each class to match the minority class
            balanced_dfs = []
            for label in class_counts.index:
                if label == min_class:
                    # Keep all samples from minority class
                    class_df = train_df[train_df['label'] == label]
                else:
                    # Undersample majority classes
                    class_df = train_df[train_df['label'] == label]
                    class_df = resample(class_df, replace=False, 
                                       n_samples=min_count, 
                                       random_state=random_state)
                balanced_dfs.append(class_df)
            
            # Combine resampled data
            balanced_df = pd.concat(balanced_dfs)
            print(f"After undersampling: {len(balanced_df)} samples")
            
            # Return equal weights since classes are now balanced
            equal_weights = {label: 1.0 for label in class_counts.index}
            return balanced_df, equal_weights
        
        elif strategy == "oversample":
            # Oversample minority classes
            from sklearn.utils import resample
            
            # Find the majority class
            max_class = class_counts.idxmax()
            max_count = class_counts[max_class]
            
            # Oversample each class to match the majority class
            balanced_dfs = []
            for label in class_counts.index:
                if label == max_class:
                    # Keep all samples from majority class
                    class_df = train_df[train_df['label'] == label]
                else:
                    # Oversample minority classes
                    class_df = train_df[train_df['label'] == label]
                    class_df = resample(class_df, replace=True, 
                                       n_samples=max_count, 
                                       random_state=random_state)
                balanced_dfs.append(class_df)
            
            # Combine resampled data
            balanced_df = pd.concat(balanced_dfs)
            print(f"After oversampling: {len(balanced_df)} samples")
            
            # Return equal weights since classes are now balanced
            equal_weights = {label: 1.0 for label in class_counts.index}
            return balanced_df, equal_weights
        
        elif strategy == "smote":
            # Use SMOTE to generate synthetic examples for minority classes
            from imblearn.over_sampling import SMOTE
            
            # Convert DataFrame to features and labels
            X = train_df['text'].fillna('').values.reshape(-1, 1)
            y = train_df['label'].values
            
            # Apply SMOTE to generate synthetic examples
            sm = SMOTE(random_state=random_state)
            
            # We can't apply SMOTE directly to text, so we'll just apply it to indices
            indices = np.arange(len(X)).reshape(-1, 1)
            indices_resampled, y_resampled = sm.fit_resample(indices, y)
            
            # Get the corresponding samples
            balanced_df = train_df.iloc[indices_resampled.flatten()].copy()
            balanced_df['label'] = y_resampled
            
            print(f"After SMOTE: {len(balanced_df)} samples")
            
            # Return equal weights since classes are now balanced
            equal_weights = {label: 1.0 for label in class_counts.index}
            return balanced_df, equal_weights
        
        else:
            print(f"Unknown strategy: {strategy}, using original data with weights")
            return train_df, class_weights
    
    def create_tfidf_features(self, train_df, val_df, test_df, max_features=5000, ngram_range=(1, 2)):
        """
        Create TF-IDF features.
        """
        print(f"Creating TF-IDF features (max_features={max_features}, ngram_range={ngram_range})")
        
        # Handle NaN values
        train_df['text'] = train_df['text'].fillna('')
        val_df['text'] = val_df['text'].fillna('')
        test_df['text'] = test_df['text'].fillna('')
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2
        )
        
        # Fit and transform
        X_train = vectorizer.fit_transform(train_df['text'])
        X_val = vectorizer.transform(val_df['text'])
        X_test = vectorizer.transform(test_df['text'])
        
        print(f"TF-IDF features - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Get labels
        y_train = train_df['label'].values if 'label' in train_df.columns else None
        y_val = val_df['label'].values if 'label' in val_df.columns else None
        
        # Feature information
        feature_names = vectorizer.get_feature_names_out()
        
        # Get feature importance for each class
        if y_train is not None:
            print("Top 10 features for each class:")
            X_train_array = X_train.toarray()
            for label in np.unique(y_train):
                # Get indices for this class
                class_indices = np.where(y_train == label)[0]
                
                # Calculate average TF-IDF score for each feature
                class_tfidf = X_train_array[class_indices].mean(axis=0)
                
                # Get top features
                top_indices = np.argsort(class_tfidf)[-10:][::-1]
                top_features = [(feature_names[i], class_tfidf[i]) for i in top_indices]
                
                print(f"  Class {label}:")
                for feature, score in top_features:
                    print(f"    {feature}: {score:.4f}")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'vectorizer': vectorizer,
            'feature_names': feature_names
        }
    
    def create_word2vec_features(self, train_df, val_df, test_df, vector_size=100, window=5, min_count=2, epochs=10):
        """
        Create Word2Vec features.
        """
        print(f"Creating Word2Vec features (vector_size={vector_size}, window={window})")
        
        # Handle NaN values
        train_df['text'] = train_df['text'].fillna('')
        val_df['text'] = val_df['text'].fillna('')
        test_df['text'] = test_df['text'].fillna('')
        
        # Tokenize texts
        def tokenize_text(text):
            return text.split()
        
        train_tokens = [tokenize_text(text) for text in train_df['text']]
        val_tokens = [tokenize_text(text) for text in val_df['text']]
        test_tokens = [tokenize_text(text) for text in test_df['text']]
        
        # Train Word2Vec model
        w2v_model = Word2Vec(
            sentences=train_tokens,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
            epochs=epochs
        )
        
        print(f"Word2Vec vocabulary size: {len(w2v_model.wv.index_to_key)}")
        
        # Create document vectors by averaging word vectors
        def get_document_vector(tokens, model, vector_size):
            vectors = [model.wv[word] for word in tokens if word in model.wv]
            if vectors:
                return np.mean(vectors, axis=0)
            else:
                return np.zeros(vector_size)
        
        # Generate document vectors
        X_train = np.array([get_document_vector(tokens, w2v_model, vector_size)
                          for tokens in tqdm(train_tokens, desc="Creating train vectors")])
        
        X_val = np.array([get_document_vector(tokens, w2v_model, vector_size)
                        for tokens in tqdm(val_tokens, desc="Creating val vectors")])
        
        X_test = np.array([get_document_vector(tokens, w2v_model, vector_size)
                         for tokens in tqdm(test_tokens, desc="Creating test vectors")])
        
        print(f"Word2Vec features - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Get labels
        y_train = train_df['label'].values if 'label' in train_df.columns else None
        y_val = val_df['label'].values if 'label' in val_df.columns else None
        
        # Print some word similarities
        print("\nSome word similarities:")
        for word in ['bullish', 'bearish', 'stock', 'market', 'up', 'down']:
            if word in w2v_model.wv:
                similar_words = w2v_model.wv.most_similar(word, topn=5)
                print(f"  {word}: {similar_words}")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'w2v_model': w2v_model,
            'train_tokens': train_tokens
        }
    
    def create_mini_sentence_bert_features(self, train_df, val_df, test_df, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """
        Create sentence embeddings using a lighter Sentence-BERT model.
        """
        print(f"Creating Mini Sentence-BERT features using {model_name}")
        
        # Handle NaN values
        train_df['text'] = train_df['text'].fillna('')
        val_df['text'] = val_df['text'].fillna('')
        test_df['text'] = test_df['text'].fillna('')
        
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        
        # Disable gradient calculations
        for param in model.parameters():
            param.requires_grad = False
        
        # Function to get mean pooling
        def mean_pooling(model_output, attention_mask):
            # Mean pooling - take average of all token embeddings
            token_embeddings = model_output.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Function to get embeddings
        def get_embeddings(texts, batch_size=32, max_length=128):
            embeddings = []
            
            for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                encoded_input = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                ).to(device)
                
                # Compute token embeddings
                with torch.no_grad():
                    model_output = model(**encoded_input)
                    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask']).cpu().numpy()
                    embeddings.append(sentence_embeddings)
            
            return np.vstack(embeddings)
        
        # Generate embeddings
        X_train = get_embeddings(train_df['text'].tolist())
        X_val = get_embeddings(val_df['text'].tolist())
        X_test = get_embeddings(test_df['text'].tolist())
        
        print(f"Mini Sentence-BERT features - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Get labels
        y_train = train_df['label'].values if 'label' in train_df.columns else None
        y_val = val_df['label'].values if 'label' in val_df.columns else None
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'tokenizer': tokenizer,
            'model': model,
            'scaler': scaler,
            'get_embeddings': get_embeddings
        }
    
    def process_features(self, preprocessing_version, imbalance_strategy="weight"):
        """
        Process all feature types for a given preprocessing version.
        
        Args:
            preprocessing_version: Name of preprocessing version
            imbalance_strategy: Strategy to handle class imbalance
            
        Returns:
            Dictionary with all features and metadata
        """
        print(f"\n{'='*80}")
        print(f"Processing features for {preprocessing_version}")
        print(f"{'='*80}")
        
        # Load data
        train_df, val_df, test_df = self.load_data(preprocessing_version)
        
        if train_df is None:
            print("Failed to load data.")
            return None
        
        # Handle class imbalance
        balanced_train_df, class_weights = self.handle_class_imbalance(
            train_df, strategy=imbalance_strategy
        )
        
        # Process all feature types
        features = {}
        
        # 1. TF-IDF Features
        print("\nProcessing TF-IDF features...")
        tfidf_features = self.create_tfidf_features(balanced_train_df, val_df, test_df)
        features['tfidf'] = tfidf_features
        
        # 2. Word2Vec Features
        print("\nProcessing Word2Vec features...")
        w2v_features = self.create_word2vec_features(balanced_train_df, val_df, test_df)
        features['word2vec'] = w2v_features
        
        # 3. Mini Sentence-BERT Features
        print("\nProcessing Mini Sentence-BERT features...")
        msbert_features = self.create_mini_sentence_bert_features(balanced_train_df, val_df, test_df)
        features['mini_sbert'] = msbert_features
        
        # Save class weights
        features['class_weights'] = class_weights
        features['preprocessing_version'] = preprocessing_version
        
        # Create output directory for this preprocessing version
        output_dir = os.path.join(self.output_dir, preprocessing_version)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save feature info for quick loading
        feature_info = {
            'tfidf_shape': tfidf_features['X_train'].shape,
            'word2vec_shape': w2v_features['X_train'].shape,
            'mini_sbert_shape': msbert_features['X_train'].shape,
            'train_samples': len(balanced_train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'class_weights': class_weights
        }
        
        
        with open(os.path.join(output_dir, 'feature_info.json'), 'w') as f:
            json.dump({k: str(v) for k, v in feature_info.items()}, f)
        
        return features
    
    def process_all_preprocessing_versions(self, imbalance_strategy="weight"):
        """
        Process features for all preprocessing versions.
        """
        all_features = {}
        
        for version in self.preprocessing_versions:
            features = self.process_features(version, imbalance_strategy)
            all_features[version] = features
        
        return all_features