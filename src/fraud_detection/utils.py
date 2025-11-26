import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import logging
from typing import Dict, List, Any

def setup_logging(log_file="pipeline.log"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def timer(func):
    """Decorator to measure execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate comprehensive evaluation metrics"""
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, precision_score, 
        recall_score, f1_score, confusion_matrix
    )
    
    metrics = {
        'auc': roc_auc_score(y_true, y_prob),
        'average_precision': average_precision_score(y_true, y_prob),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    
    return metrics

def plot_feature_importance(feature_importance, feature_names, top_n=20):
    """Plot feature importance"""
    plt.figure(figsize=(10, 8))
    indices = np.argsort(feature_importance)[-top_n:]
    
    plt.barh(range(top_n), feature_importance[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Top Feature Importance')
    plt.tight_layout()
    plt.show()

def memory_usage(df):
    """Calculate memory usage of DataFrame"""
    return df.memory_usage(deep=True).sum() / 1024**2  # MB

def reduce_memory_usage(df):
    """Reduce memory usage of DataFrame"""
    start_mem = memory_usage(df)
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = memory_usage(df)
    reduction = (start_mem - end_mem) / start_mem * 100
    print(f"Memory reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({reduction:.1f}% reduction)")
    
    return df

def validate_data_splits(X_train, X_val, y_train, y_val):
    """Validate that train/validation splits are proper"""
    assert len(X_train) == len(y_train), "Train features and labels length mismatch"
    assert len(X_val) == len(y_val), "Validation features and labels length mismatch"
    assert set(X_train.columns) == set(X_val.columns), "Train and validation feature mismatch"
    
    print("✓ Data splits validated successfully")
    print(f"  Train: {X_train.shape} features, {y_train.shape} labels")
    print(f"  Validation: {X_val.shape} features, {y_val.shape} labels")