import pandas as pd
import numpy as np
import os

class DataManager:
    def __init__(self):
        self.DATA_RAW_PATH = "../data/raw/"
        self.DATA_PROCESSED_PATH = "../data/processed/"
        self.DATA_FEATURES_PATH = "../data/processed/features/"
        
    def ensure_directories(self):
        """Ensure all required directories exist"""
        os.makedirs(self.DATA_RAW_PATH, exist_ok=True)
        os.makedirs(self.DATA_PROCESSED_PATH, exist_ok=True)
        os.makedirs(self.DATA_FEATURES_PATH, exist_ok=True)
        
    def load_raw_data(self):
        """Load raw transaction and identity data"""
        train_transaction = pd.read_csv(f"{self.DATA_RAW_PATH}train_transaction.csv")
        train_identity = pd.read_csv(f"{self.DATA_RAW_PATH}train_identity.csv")
        test_transaction = pd.read_csv(f"{self.DATA_RAW_PATH}test_transaction.csv")
        test_identity = pd.read_csv(f"{self.DATA_RAW_PATH}test_identity.csv")
        return train_transaction, train_identity, test_transaction, test_identity
    
    def save_processed_data(self, train_data, test_data, suffix=""):
        """Save processed data with optional suffix"""
        if suffix:
            train_path = f"{self.DATA_PROCESSED_PATH}train_data_processed_{suffix}.parquet"
            test_path = f"{self.DATA_PROCESSED_PATH}test_data_processed_{suffix}.parquet"
        else:
            train_path = f"{self.DATA_PROCESSED_PATH}train_data_processed.parquet"
            test_path = f"{self.DATA_PROCESSED_PATH}test_data_processed.parquet"
            
        train_data.to_parquet(train_path, index=False)
        test_data.to_parquet(test_path, index=False)
        return train_path, test_path
    
    def load_processed_data(self, suffix=""):
        """Load processed data with optional suffix"""
        if suffix:
            train_path = f"{self.DATA_PROCESSED_PATH}train_data_processed_{suffix}.parquet"
            test_path = f"{self.DATA_PROCESSED_PATH}test_data_processed_{suffix}.parquet"
        else:
            train_path = f"{self.DATA_PROCESSED_PATH}train_data_processed.parquet"
            test_path = f"{self.DATA_PROCESSED_PATH}test_data_processed.parquet"
            
        train_data = pd.read_parquet(train_path)
        test_data = pd.read_parquet(test_path)
        return train_data, test_data
    
    def save_features(self, train_data, test_data, suffix=""):
        """Save feature data with optional suffix"""
        if suffix:
            train_path = f"{self.DATA_FEATURES_PATH}train_features_{suffix}.parquet"
            test_path = f"{self.DATA_FEATURES_PATH}test_features_{suffix}.parquet"
        else:
            train_path = f"{self.DATA_FEATURES_PATH}train_features.parquet"
            test_path = f"{self.DATA_FEATURES_PATH}test_features.parquet"
            
        train_data.to_parquet(train_path, index=False)
        test_data.to_parquet(test_path, index=False)
        return train_path, test_path
    
    def load_features(self, suffix=""):
        """Load feature data with optional suffix"""
        if suffix:
            train_path = f"{self.DATA_FEATURES_PATH}train_features_{suffix}.parquet"
            test_path = f"{self.DATA_FEATURES_PATH}test_features_{suffix}.parquet"
        else:
            train_path = f"{self.DATA_FEATURES_PATH}train_features.parquet"
            test_path = f"{self.DATA_FEATURES_PATH}test_features.parquet"
            
        train_data = pd.read_parquet(train_path)
        test_data = pd.read_parquet(test_path)
        return train_data, test_data