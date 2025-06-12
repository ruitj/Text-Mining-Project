import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def analyze_preprocessed_data(train_df, val_df, test_df, versions):
    """
    Analyze preprocessed data versions for missing values and visualize with word clouds.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        versions: List of preprocessing version names
    """
    print("\n===== PREPROCESSED DATA ANALYSIS =====\n")
    
    # Check for missing values in all datasets
    print("\nProcessed versions missing values:")
    for version in versions:
        col_name = f'text_processed_{version}'
        train_missing = train_df[col_name].isna().sum()
        val_missing = val_df[col_name].isna().sum()
        test_missing = test_df[col_name].isna().sum()
        
        print(f"  {version} - Train: {train_missing}, Val: {val_missing}, Test: {test_missing}")
    
    # Generate word clouds for each version
    print("\nGenerating word clouds...")
    plt.figure(figsize=(20, 15))
    
    for i, version in enumerate(versions):
        col_name = f'text_processed_{version}'
        
        # Combine text from train set
        all_text = ' '.join(train_df[col_name].fillna('').astype(str))
        
        plt.subplot(2, 3, i + 1)
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                              max_words=100, contour_width=3, contour_color='steelblue')
        wordcloud.generate(all_text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud - {version}', fontsize=16)
    
    plt.tight_layout()
    plt.show()
    
    # Optionally, generate separate word clouds for test data
    print("\nGenerating test data word clouds...")
    plt.figure(figsize=(20, 15))
    
    for i, version in enumerate(versions):
        col_name = f'text_processed_{version}'
        
        # Combine text from test set
        test_text = ' '.join(test_df[col_name].fillna('').astype(str))
        
        plt.subplot(2, 3, i + 1)
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                              max_words=100, contour_width=3, contour_color='steelblue')
        wordcloud.generate(test_text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Test Word Cloud - {version}', fontsize=16)
    
    plt.tight_layout()
    plt.show()
    
    # Compare token counts across datasets
    print("\nToken count statistics across datasets:")
    for version in versions:
        col_name = f'text_processed_{version}'
        
        # Calculate token counts
        train_df[f'token_count_{version}'] = train_df[col_name].apply(lambda x: len(str(x).split()))
        val_df[f'token_count_{version}'] = val_df[col_name].apply(lambda x: len(str(x).split()))
        test_df[f'token_count_{version}'] = test_df[col_name].apply(lambda x: len(str(x).split()))
        
        # Calculate statistics
        train_avg = train_df[f'token_count_{version}'].mean()
        val_avg = val_df[f'token_count_{version}'].mean()
        test_avg = test_df[f'token_count_{version}'].mean()
        
        print(f"  {version} - Average tokens:")
        print(f"    Train: {train_avg:.2f}")
        print(f"    Val: {val_avg:.2f}")
        print(f"    Test: {test_avg:.2f}")
        

def export_processed_data(train_df, val_df, test_df, train_processed, val_processed, test_processed, base_dir="processed_data"):
    """
    Export preprocessed data to CSV files and organize into folders.
    
    Args:
        train_df: Original training DataFrame
        val_df: Original validation DataFrame
        test_df: Original test DataFrame
        train_processed: Dictionary with processed text for training set
        val_processed: Dictionary with processed text for validation set
        test_processed: Dictionary with processed text for test set
        base_dir: Base directory for creating folders
    """
    import os
    
    # Create base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Created base directory: {base_dir}")
    
    # For each preprocessing strategy
    for version in train_processed.keys():
        # Create directory for this version
        version_dir = os.path.join(base_dir, version)
        if not os.path.exists(version_dir):
            os.makedirs(version_dir)
            print(f"Created directory: {version_dir}")
        
        # Create copies of the original DataFrames
        train_copy = train_df.copy()
        val_copy = val_df.copy()
        test_copy = test_df.copy()
        
        # Replace the 'text' column with the processed text
        train_copy['text'] = train_processed[version]
        val_copy['text'] = val_processed[version]
        test_copy['text'] = test_processed[version]
        
        # Export to CSV files
        train_file = os.path.join(version_dir, 'train.csv')
        val_file = os.path.join(version_dir, 'val.csv')
        test_file = os.path.join(version_dir, 'test.csv')
        
        train_copy.to_csv(train_file, index=False)
        val_copy.to_csv(val_file, index=False)
        test_copy.to_csv(test_file, index=False)
        
        print(f"Exported {version} - Train: {train_file}, Val: {val_file}, Test: {test_file}")
    
    print("\nData export complete. Directory structure:")
    for version in train_processed.keys():
        print(f"{base_dir}/{version}/")
        print(f"├── train.csv")
        print(f"├── val.csv")
        print(f"└── test.csv")
