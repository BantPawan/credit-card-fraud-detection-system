import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import os

class FeatureSelection:
    def __init__(self):
        self.DATA_FEATURES_PATH = "../data/processed/features/"
        
    def load_engineered_data(self):
        """Load fully engineered features"""
        print("Loading engineered features...")
        train_data = pd.read_parquet(f"{self.DATA_FEATURES_PATH}train_features_fully_engineered.parquet")
        test_data = pd.read_parquet(f"{self.DATA_FEATURES_PATH}test_features_fully_engineered.parquet")
        
        print(f"Train shape: {train_data.shape}")
        print(f"Test shape: {test_data.shape}")
        return train_data, test_data
    
    def remove_low_importance_features(self, train_data, test_data, threshold='median'):
        """Remove low importance features using Random Forest"""
        print("Selecting features based on importance...")
        
        # Prepare data
        X = train_data.drop(columns=['TransactionID', 'isFraud'])
        y = train_data['isFraud']
        
        # Split for feature selection
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest for feature importance
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=50,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X_train, y_train)
        
        # Select features based on importance
        selector = SelectFromModel(rf, threshold=threshold, prefit=True)
        selected_features = X.columns[selector.get_support()].tolist()
        
        print(f"Selected {len(selected_features)} features from {len(X.columns)} total features")
        
        # Apply selection
        train_selected = train_data[['TransactionID', 'isFraud'] + selected_features].copy()
        test_selected = test_data[['TransactionID'] + selected_features].copy()
        
        return train_selected, test_selected, selected_features
    
    def analyze_feature_correlation(self, train_data, threshold=0.95):
        """Remove highly correlated features"""
        print("Analyzing feature correlations...")
        
        # Get numerical features only
        numerical_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col not in ['TransactionID', 'isFraud']]
        
        if not numerical_cols:
            return train_data, test_data
            
        # Calculate correlation matrix
        corr_matrix = train_data[numerical_cols].corr().abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        print(f"Removing {len(to_drop)} highly correlated features")
        
        # Remove correlated features
        train_reduced = train_data.drop(columns=to_drop)
        test_reduced = test_data.drop(columns=[col for col in to_drop if col in test_data.columns])
        
        return train_reduced, test_reduced
    
    def select_features_by_missingness(self, train_data, test_data, missing_threshold=0.8):
        """Remove features with high missingness"""
        print("Removing features with high missingness...")
        
        missing_percentage = train_data.isnull().mean()
        high_missing_features = missing_percentage[missing_percentage > missing_threshold].index.tolist()
        
        # Don't remove target or ID columns
        high_missing_features = [f for f in high_missing_features if f not in ['TransactionID', 'isFraud']]
        
        print(f"Removing {len(high_missing_features)} features with >{missing_threshold*100}% missing values")
        
        train_reduced = train_data.drop(columns=high_missing_features)
        test_reduced = test_data.drop(columns=[col for col in high_missing_features if col in test_data.columns])
        
        return train_reduced, test_reduced
    
    def validate_feature_sets(self, train_data, test_data):
        """Validate that feature sets are consistent"""
        train_features = set(train_data.columns) - {'TransactionID', 'isFraud'}
        test_features = set(test_data.columns) - {'TransactionID'}
        
        missing_in_test = train_features - test_features
        missing_in_train = test_features - train_features
        
        if missing_in_test:
            print(f"WARNING: {len(missing_in_test)} features missing in test data")
        if missing_in_train:
            print(f"WARNING: {len(missing_in_train)} features missing in train data")
        
        if not missing_in_test and not missing_in_train:
            print("✓ All features consistent between train and test")
        
        return list(train_features & test_features)
    
    def save_selected_features(self, train_data, test_data):
        """Save feature-selected datasets"""
        train_data.to_parquet(f"{self.DATA_FEATURES_PATH}train_features_selected_enhanced.parquet", index=False)
        test_data.to_parquet(f"{self.DATA_FEATURES_PATH}test_features_selected_enhanced.parquet", index=False)
        
        print("Saved selected features:")
        print(f"Train: {self.DATA_FEATURES_PATH}train_features_selected_enhanced.parquet")
        print(f"Test: {self.DATA_FEATURES_PATH}test_features_selected_enhanced.parquet")
        print(f"Final shapes - Train: {train_data.shape}, Test: {test_data.shape}")
    
    def run(self):
        """Execute complete feature selection pipeline"""
        print("=== STARTING FEATURE SELECTION ===")
        
        # Load engineered data
        train_data, test_data = self.load_engineered_data()
        
        # Remove features with high missingness
        train_reduced, test_reduced = self.select_features_by_missingness(train_data, test_data, missing_threshold=0.8)
        
        # Remove highly correlated features
        train_reduced, test_reduced = self.analyze_feature_correlation(train_reduced, threshold=0.95)
        
        # Select features by importance
        train_selected, test_selected, selected_features = self.remove_low_importance_features(train_reduced, test_reduced)
        
        # Validate feature sets
        common_features = self.validate_feature_sets(train_selected, test_selected)
        
        print(f"\nFeature Selection Summary:")
        print(f"Original features: {train_data.shape[1] - 2}")  # Exclude TransactionID and isFraud
        print(f"Selected features: {len(common_features)}")
        print(f"Reduction: {((train_data.shape[1] - 2 - len(common_features)) / (train_data.shape[1] - 2) * 100):.1f}%")
        
        # Save selected features
        self.save_selected_features(train_selected, test_selected)
        
        print("=== FEATURE SELECTION COMPLETED SUCCESSFULLY! ===")

if __name__ == "__main__":
    feature_sel = FeatureSelection()
    feature_sel.run()