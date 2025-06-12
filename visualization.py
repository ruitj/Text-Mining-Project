# visualizations.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc

def log_results(results_list, log_file="model_results_log.csv"):
    """
    Log results to CSV file as models are trained.
    
    Args:
        results_list: List of dictionaries with model results
        log_file: Path to CSV file for logging
        
    Returns:
        DataFrame with all results
    """
    df = pd.DataFrame(results_list)
    df.to_csv(log_file, index=False)
    return df

def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=True, title="Confusion Matrix", 
                         output_path=None, figsize=(10, 8)):
    """
    Create and optionally save a confusion matrix visualization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names (optional)
        normalize: Whether to normalize the confusion matrix
        title: Plot title
        output_path: Path to save the figure (if None, just displays)
        figsize: Figure size as (width, height)
    """
    plt.figure(figsize=figsize)
    
    if normalize:
        # Use 'all' instead of 'true' for normalization
        cm = pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], normalize='all')
        fmt = '.2f'
    else:
        cm = pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'])
        fmt = 'd'
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues')
    plt.title(title)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def plot_model_comparison(results_df, metric='Val F1', group_by='Preprocessing', 
                          hue='Feature Type', title=None, output_path=None):
    """
    Create bar plot comparing model performance across different configurations.
    
    Args:
        results_df: DataFrame with model results
        metric: Metric to plot (e.g., 'Val F1', 'Val Accuracy')
        group_by: Column to group by on x-axis
        hue: Column to use for color grouping
        title: Plot title
        output_path: Path to save the figure
    """
    plt.figure(figsize=(12, 8))
    
    # Create the bar plot
    ax = sns.barplot(data=results_df, x=group_by, y=metric, hue=hue)
    
    # Set title
    if title:
        plt.title(title)
    else:
        plt.title(f'{metric} by {group_by} and {hue}')
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45 if len(results_df[group_by].unique()) > 4 else 0)
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def plot_parameter_effect(results_df, param_name, metric='Val F1', hue=None, 
                          title=None, output_path=None):
    """
    Create box plot showing effect of a parameter on model performance.
    
    Args:
        results_df: DataFrame with model results
        param_name: Parameter to analyze
        metric: Metric to plot on y-axis
        hue: Column to use for color grouping
        title: Plot title
        output_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    # Create the box plot
    sns.boxplot(data=results_df, x=param_name, y=metric, hue=hue)
    
    # Set title
    if title:
        plt.title(title)
    else:
        plt.title(f'Effect of {param_name} on {metric}')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def plot_overfitting_analysis(results_df, group_by='Feature Type', title=None, output_path=None):
    """
    Create visualization showing overfitting by a grouping variable.
    
    Args:
        results_df: DataFrame with model results (must have 'Overfitting' column)
        group_by: Column to group by on x-axis
        title: Plot title
        output_path: Path to save the figure
    """
    if 'Overfitting' not in results_df.columns and 'Train F1' in results_df.columns and 'Val F1' in results_df.columns:
        # Calculate overfitting if not already present
        results_df['Overfitting'] = results_df['Train F1'] - results_df['Val F1']
    
    plt.figure(figsize=(10, 6))
    
    # Create the box plot
    sns.boxplot(data=results_df, x=group_by, y='Overfitting')
    
    # Add a horizontal line at 0 (no overfitting)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    
    # Set title
    if title:
        plt.title(title)
    else:
        plt.title(f'Overfitting by {group_by}')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def plot_roc_curves(y_true, y_score, class_names=None, title=None, output_path=None):
    """
    Create ROC curves for multiclass classification.
    
    Args:
        y_true: True labels
        y_score: Prediction probabilities (shape: n_samples, n_classes)
        class_names: Names for each class
        title: Plot title
        output_path: Path to save the figure
    """
    n_classes = y_score.shape[1]
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        y_true_binary = (y_true == i).astype(int)
        fpr[i], tpr[i], _ = roc_curve(y_true_binary, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Generate class names if not provided
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title or 'Receiver Operating Characteristic Curves')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def find_best_model(results_df, metric='Val F1', min_columns=None):
    """
    Find the best model based on a specific metric and display its details.
    
    Args:
        results_df: DataFrame with model results
        metric: Metric to use for ranking
        min_columns: List of minimum columns to display (will include all if None)
        
    Returns:
        Series with the best model's details
    """
    # Sort by the specified metric
    sorted_df = results_df.sort_values(metric, ascending=False)
    
    # Get the best model
    best_model = sorted_df.iloc[0]
    
    # Display details
    print(f"BEST MODEL BY {metric}:")
    print("=" * 50)
    
    if min_columns:
        for col in min_columns:
            if col in best_model:
                print(f"{col}: {best_model[col]}")
    else:
        for col, value in best_model.items():
            print(f"{col}: {value}")
    
    return best_model

def create_results_summary(results_dir="model_results", output_file="all_models_summary.csv"):
    """
    Create a comprehensive summary of all models by scanning the results directory.
    
    Args:
        results_dir: Base directory with model results
        output_file: File to save the summary
        
    Returns:
        DataFrame with model summaries
    """
    summary_rows = []
    
    # Scan through the directory structure
    for preprocessing in os.listdir(results_dir):
        prep_dir = os.path.join(results_dir, preprocessing)
        if not os.path.isdir(prep_dir):
            continue
            
        for feature_type in os.listdir(prep_dir):
            feat_dir = os.path.join(prep_dir, feature_type)
            if not os.path.isdir(feat_dir):
                continue
                
            for model_name in os.listdir(feat_dir):
                model_dir = os.path.join(feat_dir, model_name)
                if not os.path.isdir(model_dir):
                    continue
                    
                # Check for metrics file
                metrics_file = os.path.join(model_dir, 'metrics.json')
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        import json
                        metrics = json.load(f)
                        
                        # Add directory info if not in metrics
                        if 'preprocessing_version' not in metrics:
                            metrics['preprocessing_version'] = preprocessing
                        if 'feature_type' not in metrics:
                            metrics['feature_type'] = feature_type
                        if 'model_name' not in metrics:
                            metrics['model_name'] = model_name
                            
                        summary_rows.append(metrics)
    
    # Create DataFrame from collected data
    summary_df = pd.DataFrame(summary_rows)
    
    # Save to CSV
    if len(summary_df) > 0:
        summary_df.to_csv(output_file, index=False)
        print(f"Saved comprehensive summary to {output_file}")
    else:
        print("No model results found to summarize.")
    
    return summary_df