import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os

class DataPreparation:
    def __init__(self):
        self.DATA_RAW_PATH = "../data/raw/"
        self.DATA_PROCESSED_PATH = "../data/processed/"
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        
    def load_data(self):
        """Load raw transaction and identity data"""
        print("Loading raw data...")
        
        train_transaction = pd.read_csv(f"{self.DATA_RAW_PATH}train_transaction.csv")
        train_identity = pd.read_csv(f"{self.DATA_RAW_PATH}train_identity.csv")
        test_transaction = pd.read_csv(f"{self.DATA_RAW_PATH}test_transaction.csv")
        test_identity = pd.read_csv(f"{self.DATA_RAW_PATH}test_identity.csv")
        
        print(f"Train Transaction: {train_transaction.shape}")
        print(f"Train Identity: {train_identity.shape}")
        print(f"Test Transaction: {test_transaction.shape}")
        print(f"Test Identity: {test_identity.shape}")
        
        return train_transaction, train_identity, test_transaction, test_identity
    
    def merge_datasets(self, train_transaction, train_identity, test_transaction, test_identity):
        """Merge transaction and identity data"""
        print("Merging datasets...")
        
        # Merge train data
        train_data = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
        print(f"Merged train data shape: {train_data.shape}")
        
        # Merge test data
        test_data = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')
        print(f"Merged test data shape: {test_data.shape}")
        
        return train_data, test_data
    
    def remove_constant_columns(self, train_data, test_data):
        """Remove columns with zero variance"""
        constant_cols_train = [col for col in train_data.columns if train_data[col].nunique() <= 1]
        constant_cols_test = [col for col in test_data.columns if test_data[col].nunique() <= 1]
        
        print(f"Constant columns in train: {len(constant_cols_train)}")
        print(f"Constant columns in test: {len(constant_cols_test)}")
        
        if constant_cols_train:
            train_data = train_data.drop(columns=constant_cols_train)
            test_data = test_data.drop(columns=[col for col in constant_cols_train if col in test_data.columns])
            print("Removed constant columns from both datasets")
        
        return train_data, test_data
    
    def analyze_missingness(self, df, dataset_name):
        """Comprehensive missing value analysis"""
        missing_data = df.isnull().sum()
        missing_percentage = (missing_data / len(df)) * 100
        
        print(f"\n{dataset_name} - Missing Value Summary:")
        print(f"Total columns: {df.shape[1]}")
        print(f"Columns with >90% missing: {(missing_percentage > 90).sum()}")
        print(f"Columns with >80% missing: {(missing_percentage > 80).sum()}") 
        print(f"Columns with >50% missing: {(missing_percentage > 50).sum()}")
        print(f"Columns with >20% missing: {(missing_percentage > 20).sum()}")
        
        # Top 10 most missing columns
        top_missing = missing_percentage.sort_values(ascending=False).head(10)
        print(f"\nTop 10 most missing columns:")
        for col, pct in top_missing.items():
            print(f"  {col}: {pct:.1f}%")
        
        return missing_percentage
    
    def analyze_target(self, train_data):
        """Analyze target variable distribution"""
        if 'isFraud' in train_data.columns:
            fraud_stats = train_data['isFraud'].value_counts()
            fraud_percentage = train_data['isFraud'].value_counts(normalize=True) * 100
            
            print("\n=== FRAUD DISTRIBUTION ===")
            print(f"Non-Fraud (0): {fraud_stats[0]:,} transactions ({fraud_percentage[0]:.2f}%)")
            print(f"Fraud (1): {fraud_stats[1]:,} transactions ({fraud_percentage[1]:.2f}%)")
            print(f"Imbalance Ratio: {fraud_stats[0]/fraud_stats[1]:.1f}:1")
            
            # Missing value analysis by target
            fraud_missing_avg = train_data[train_data['isFraud'] == 1].isnull().mean().mean() * 100
            non_fraud_missing_avg = train_data[train_data['isFraud'] == 0].isnull().mean().mean() * 100
            
            print(f"\nMissing Value Pattern by Fraud Status:")
            print(f"Average missing values in Fraud transactions: {fraud_missing_avg:.1f}%")
            print(f"Average missing values in Non-Fraud transactions: {non_fraud_missing_avg:.1f}%")
            print(f"Difference: {abs(fraud_missing_avg - non_fraud_missing_avg):.1f}%")
            
            return fraud_percentage
        else:
            print("Target variable 'isFraud' not found in training data!")
            return None
    
    def save_data(self, train_data, test_data):
        """Save processed data"""
        os.makedirs(self.DATA_PROCESSED_PATH, exist_ok=True)
        
        train_data.to_parquet(f"{self.DATA_PROCESSED_PATH}train_data_processed.parquet", index=False)
        test_data.to_parquet(f"{self.DATA_PROCESSED_PATH}test_data_processed.parquet", index=False)
        
        print("\nSaved processed data:")
        print(f"Train: {self.DATA_PROCESSED_PATH}train_data_processed.parquet")
        print(f"Test: {self.DATA_PROCESSED_PATH}test_data_processed.parquet")
        
        print(f"\nFinal dataset shapes:")
        print(f"Train: {train_data.shape}")
        print(f"Test: {test_data.shape}")
    
    def run(self):
        """Execute complete data preparation pipeline"""
        print("=== STARTING DATA PREPARATION ===")
        
        # Load data
        train_transaction, train_identity, test_transaction, test_identity = self.load_data()
        
        # Merge datasets
        train_data, test_data = self.merge_datasets(train_transaction, train_identity, test_transaction, test_identity)
        
        # Remove constant columns
        train_data, test_data = self.remove_constant_columns(train_data, test_data)
        
        # Analyze missing values
        train_missing_pct = self.analyze_missingness(train_data, "Train Data")
        test_missing_pct = self.analyze_missingness(test_data, "Test Data")
        
        # Analyze target variable
        fraud_percentage = self.analyze_target(train_data)
        
        # Save processed data
        self.save_data(train_data, test_data)
        
        print("\n=== DATA PREPARATION COMPLETED SUCCESSFULLY! ===")

if __name__ == "__main__":
    data_prep = DataPreparation()
    data_prep.run()