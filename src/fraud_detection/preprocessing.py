import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class Preprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.frequency_encoders = {}
        
    def create_preprocessing_pipeline(self, cont_cols, bin_cols, cat_cols):
        """Create preprocessing pipeline for different feature types"""
        transformers = []
        
        if cont_cols:
            transformers.append(("num", StandardScaler(), cont_cols))
        
        if bin_cols:
            transformers.append(("binary", "passthrough", bin_cols))
            
        if cat_cols:
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', 'passthrough')  # Encoding handled separately
            ])
            transformers.append(("cat", cat_pipeline, cat_cols))
        
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        return preprocessor
    
    def encode_categorical_features(self, df, categorical_cols):
        """Encode categorical features using appropriate strategies"""
        for col in categorical_cols:
            if col not in ['TransactionID', 'isFraud']:
                n_unique = df[col].nunique()
                
                if n_unique <= 10:
                    # Low cardinality - label encoding
                    if col not in self.label_encoders:
                        le = LabelEncoder()
                        unique_vals = df[col].fillna('unknown').unique()
                        le.fit(unique_vals)
                        self.label_encoders[col] = le
                    
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].fillna('unknown'))
                    
                else:
                    # High cardinality - frequency encoding
                    if col not in self.frequency_encoders:
                        freq_map = df[col].value_counts().to_dict()
                        self.frequency_encoders[col] = freq_map
                    
                    df[f'{col}_freq'] = df[col].map(self.frequency_encoders[col])
                    df[f'{col}_freq'] = df[f'{col}_freq'].fillna(1)
        
        return df
    
    def handle_missing_values(self, df, strategy='constant', fill_value=-999):
        """Handle missing values in the dataset"""
        # Separate numerical and categorical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Handle numerical missing values
        if strategy == 'constant':
            df[numerical_cols] = df[numerical_cols].fillna(fill_value)
        elif strategy == 'median':
            for col in numerical_cols:
                df[col] = df[col].fillna(df[col].median())
        
        # Handle categorical missing values
        df[categorical_cols] = df[categorical_cols].fillna('missing')
        
        return df
    
    def detect_feature_types(self, df):
        """Detect continuous, binary, and categorical features"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target and ID columns
        numerical_cols = [col for col in numerical_cols if col not in ['TransactionID', 'isFraud']]
        categorical_cols = [col for col in categorical_cols if col not in ['TransactionID', 'isFraud']]
        
        # Identify binary features
        binary_cols = [col for col in numerical_cols if df[col].nunique() == 2]
        continuous_cols = [col for col in numerical_cols if col not in binary_cols]
        
        return continuous_cols, binary_cols, categorical_cols