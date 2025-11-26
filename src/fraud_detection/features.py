import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineering:
    def __init__(self):
        self.DATA_PROCESSED_PATH = "../data/processed/"
        self.DATA_FEATURES_PATH = "../data/processed/features/"
        os.makedirs(self.DATA_FEATURES_PATH, exist_ok=True)
        
    def load_processed_data(self):
        """Load data from data preparation step"""
        print("Loading processed data...")
        train_data = pd.read_parquet(f"{self.DATA_PROCESSED_PATH}train_data_processed.parquet")
        test_data = pd.read_parquet(f"{self.DATA_PROCESSED_PATH}test_data_processed.parquet")
        
        print(f"Train shape: {train_data.shape}")
        print(f"Test shape: {test_data.shape}")
        return train_data, test_data
    
    def strategic_feature_selection(self, train_data, test_data):
        """Select high-value features based on domain knowledge"""
        def add_if_exists(feature_list, df):
            return [f for f in feature_list if f in df.columns]
        
        # Core feature groups
        core_features = ['TransactionID', 'TransactionDT', 'TransactionAmt', 'ProductCD', 'isFraud']
        card_features = [f'card{i}' for i in range(1, 7)]
        count_features = [f'C{i}' for i in range(1, 15)]
        time_delta_keep = ['D1', 'D2', 'D3', 'D4', 'D8', 'D10', 'D15']
        match_keep = [f'M{i}' for i in range(1, 7)]
        address_features = ['addr1', 'addr2']
        email_features = ['P_emaildomain', 'R_emaildomain']
        identity_numeric_keep = [f'id_{i:02d}' for i in range(1, 12)]
        identity_categorical_keep = ['id_30','id_31','id_33','id_34','id_35','id_36','id_37','id_38']
        device_features = ['DeviceType', 'DeviceInfo']
        
        # Combine all groups
        all_feature_groups = (
            core_features + card_features + count_features + time_delta_keep +
            match_keep + address_features + email_features + identity_numeric_keep +
            identity_categorical_keep + device_features
        )
        
        features_to_keep = set(add_if_exists(all_feature_groups, train_data))
        print(f"Selected {len(features_to_keep)} core features to keep.")
        
        # Select top V-features by correlation
        top_v_features = self.select_top_v_features(train_data, top_n=150, missing_threshold=0.9)
        
        # Combine features
        features_common = [f for f in features_to_keep if f in train_data.columns and f in test_data.columns]
        selected_features = features_common + top_v_features
        
        # Apply selection
        train_data_selected = train_data[selected_features + ['isFraud']].copy()
        test_data_selected = test_data[selected_features].copy()
        
        print(f"After selection - Train: {train_data_selected.shape}, Test: {test_data_selected.shape}")
        return train_data_selected, test_data_selected
    
    def select_top_v_features(self, train_df, top_n=150, missing_threshold=0.9):
        """Select top V features by correlation with target"""
        v_cols = [col for col in train_df.columns if col.startswith('V')]
        
        if not v_cols or 'isFraud' not in train_df.columns:
            return v_cols
        
        # Calculate correlation with target
        v_corr = train_df[v_cols + ['isFraud']].corr()['isFraud'].drop('isFraud', errors='ignore').abs()
        
        # Filter out features with too many missing values
        missing_pct = train_df[v_cols].isnull().mean()
        valid_v_cols = missing_pct[missing_pct < missing_threshold].index
        
        # Get top correlated features
        top_v_features = v_corr.loc[valid_v_cols].sort_values(ascending=False).head(top_n).index.tolist()
        
        print(f"Selected {len(top_v_features)} V features (top {top_n} by correlation, <{missing_threshold*100}% missing)")
        if top_v_features:
            print(f"Top 5 V features: {top_v_features[:5]}")
        
        return top_v_features
    
    def create_transaction_amount_features(self, df, train_df=None):
        """Engineer transaction amount features"""
        # Use training stats if provided (for test data)
        if train_df is not None:
            amount_stats = {
                'median': train_df['TransactionAmt'].median(),
                'q75': train_df['TransactionAmt'].quantile(0.75),
                'q25': train_df['TransactionAmt'].quantile(0.25),
                'mean': train_df['TransactionAmt'].mean(),
                'std': train_df['TransactionAmt'].std()
            }
            amount_stats['iqr'] = amount_stats['q75'] - amount_stats['q25']
        else:
            amount_stats = {
                'median': df['TransactionAmt'].median(),
                'q75': df['TransactionAmt'].quantile(0.75),
                'q25': df['TransactionAmt'].quantile(0.25),
                'mean': df['TransactionAmt'].mean(),
                'std': df['TransactionAmt'].std()
            }
            amount_stats['iqr'] = amount_stats['q75'] - amount_stats['q25']
        
        # 1. Log transform
        df['TransactionAmt_log'] = np.log1p(df['TransactionAmt'])
        
        # 2. IQR-based outlier detection
        lower_bound = amount_stats['q25'] - 1.5 * amount_stats['iqr']
        upper_bound = amount_stats['q75'] + 1.5 * amount_stats['iqr']
        df['is_amount_outlier'] = ((df['TransactionAmt'] < lower_bound) | 
                                  (df['TransactionAmt'] > upper_bound)).astype(int)
        
        # 3. Amount ratios
        df['amount_to_median'] = df['TransactionAmt'] / amount_stats['median']
        df['amount_to_q75'] = df['TransactionAmt'] / amount_stats['q75']
        
        # 4. Statistical features
        df['amount_zscore'] = (df['TransactionAmt'] - amount_stats['mean']) / amount_stats['std']
        
        return df
    
    def create_temporal_features(self, df):
        """Engineer temporal features from TransactionDT"""
        # Convert seconds to datetime-like features
        df['hour'] = (df['TransactionDT'] // 3600) % 24
        df['dayofweek'] = (df['TransactionDT'] // (3600 * 24)) % 7
        
        # Fraud patterns
        df['is_fraud_peak_hour'] = (((df['hour'] >= 1) & (df['hour'] <= 4)) | 
                                   ((df['hour'] >= 15) & (df['hour'] <= 17))).astype(int)
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        # High-risk combinations
        df['is_late_night_weekend'] = ((df['is_night'] == 1) & (df['is_weekend'] == 1)).astype(int)
        
        return df
    
    def create_product_features(self, df):
        """Engineer product-related features"""
        # Risk-based product flags
        high_risk_products = ['C', 'H']
        df['is_high_risk_product'] = df['ProductCD'].isin(high_risk_products).astype(int)
        
        medium_risk_products = ['R', 'S']
        df['is_medium_risk_product'] = df['ProductCD'].isin(medium_risk_products).astype(int)
        
        # Product risk scores
        product_risk_scores = {
            'C': 3, 'H': 2, 'R': 1, 'S': 1, 'W': 0
        }
        df['product_risk_score'] = df['ProductCD'].map(product_risk_scores).fillna(0)
        
        return df
    
    def create_card_features(self, df, train_df=None):
        """Engineer card-related features"""
        # Card frequency features
        if train_df is not None:
            card1_freq = train_df['card1'].value_counts()
        else:
            card1_freq = df['card1'].value_counts()
        
        df['card1_frequency'] = df['card1'].map(card1_freq).fillna(1)
        df['is_single_use_card'] = (df['card1_frequency'] == 1).astype(int)
        df['is_high_freq_card'] = (df['card1_frequency'] > 10).astype(int)
        
        # Card type risk flags
        if 'card4' in df.columns:
            df['is_high_risk_card4'] = (df['card4'] == 'discover').astype(int)
            df['is_medium_risk_card4'] = df['card4'].isin(['mastercard', 'visa']).astype(int)
        
        # Card type flags
        if 'card6' in df.columns:
            df['is_credit_card'] = (df['card6'] == 'credit').astype(int)
            df['is_debit_card'] = (df['card6'] == 'debit').astype(int)
            df['is_debit_or_credit'] = df['card6'].isin(['credit', 'debit']).astype(int)
        
        return df
    
    def create_email_features(self, df, train_df=None):
        """Engineer email domain features"""
        # Identify high-risk domains
        high_risk_domains = set()
        
        if train_df is not None and 'isFraud' in train_df.columns:
            for email_col in ['P_emaildomain', 'R_emaildomain']:
                if email_col in train_df.columns:
                    fraud_rates = train_df.groupby(email_col)['isFraud'].mean()
                    high_risk = fraud_rates[fraud_rates > train_df['isFraud'].mean() * 2].index
                    high_risk_domains.update(high_risk)
        else:
            high_risk_domains = {'protonmail.com', 'mail.com', 'outlook.es', 'netzero.net', 'aim.com'}
        
        # Email provider extraction
        def extract_email_provider(email_domain):
            if pd.isna(email_domain): return "missing"
            email_domain = str(email_domain).lower()
            if 'gmail' in email_domain: return 'gmail'
            if 'yahoo' in email_domain: return 'yahoo' 
            if 'hotmail' in email_domain: return 'hotmail'
            if 'protonmail' in email_domain: return 'protonmail'
            if 'outlook' in email_domain: return 'outlook'
            if 'aol' in email_domain: return 'aol'
            if 'icloud' in email_domain: return 'icloud'
            return 'other'
        
        for email_col in ['P_emaildomain', 'R_emaildomain']:
            if email_col in df.columns:
                col_suffix = email_col.replace('emaildomain', '').replace('_', '')
                
                df[f'email_provider{col_suffix}'] = df[email_col].apply(extract_email_provider)
                df[f'is_high_risk{col_suffix}'] = df[email_col].isin(high_risk_domains).astype(int)
                df[f'is_protonmail{col_suffix}'] = df[email_col].str.contains('protonmail', na=False).astype(int)
                df[f'is_mail_com{col_suffix}'] = df[email_col].str.contains('mail.com', na=False).astype(int)
                df[f'is_missing{col_suffix}'] = df[email_col].isna().astype(int)
                
                popular_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'aol.com', 'icloud.com']
                df[f'is_popular{col_suffix}'] = df[email_col].isin(popular_domains).astype(int)
                df[f'is_anonymous{col_suffix}'] = (df[email_col] == 'anonymous.com').astype(int)
        
        # Email match flag
        if 'P_emaildomain' in df.columns and 'R_emaildomain' in df.columns:
            df['email_match'] = (df['P_emaildomain'] == df['R_emaildomain']).astype(int)
            df['email_match'] = df['email_match'].fillna(0)
        
        return df
    
    def create_identity_features(self, df):
        """Engineer identity and device features"""
        # Clean OS information
        def clean_os(x):
            if pd.isna(x): return "Missing"
            x = str(x).split()[0]
            if x in ['Android', 'iOS', 'Mac', 'Windows', 'Linux']: return x
            return "Other"
        
        # Clean Browser information  
        def clean_browser(x):
            if pd.isna(x): return "Missing"
            x = str(x).lower()
            if 'chrome' in x: return 'Chrome'
            if 'safari' in x: return 'Safari' 
            if 'firefox' in x: return 'Firefox'
            if 'samsung' in x or 'samsungbrowser' in x: return 'Samsung'
            if 'edge' in x or 'ie ' in x: return 'IE/Edge'
            if 'android' in x: return 'Android Browser'
            return "Other"
        
        # OS features
        if 'id_30' in df.columns:
            df['OS'] = df['id_30'].apply(clean_os)
            df['is_high_risk_OS'] = df['OS'].isin(['Other', 'Android', 'Linux']).astype(int)
            df['is_windows_OS'] = (df['OS'] == 'Windows').astype(int)
            df['is_mobile_OS'] = df['OS'].isin(['Android', 'iOS']).astype(int)
            df['OS_missing'] = df['id_30'].isna().astype(int)
        
        # Browser features
        if 'id_31' in df.columns:
            df['Browser'] = df['id_31'].apply(clean_browser)
            df['is_high_risk_browser'] = df['Browser'].isin(['Other', 'Android Browser']).astype(int)
            df['is_chrome_browser'] = (df['Browser'] == 'Chrome').astype(int)
            df['is_safari_browser'] = (df['Browser'] == 'Safari').astype(int)
            df['browser_missing'] = df['id_31'].isna().astype(int)
        
        # Device type features
        if 'DeviceType' in df.columns:
            df['is_mobile_device'] = (df['DeviceType'] == 'mobile').astype(int)
            df['is_desktop_device'] = (df['DeviceType'] == 'desktop').astype(int)
            df['DeviceType_missing'] = df['DeviceType'].isna().astype(int)
        
        # Boolean identity flags
        for bool_col in ['id_35', 'id_36', 'id_37', 'id_38']:
            if bool_col in df.columns:
                df[f'{bool_col}_is_F'] = (df[bool_col] == 'F').astype(int)
                df[f'{bool_col}_is_T'] = (df[bool_col] == 'T').astype(int)
                df[f'{bool_col}_missing'] = df[bool_col].isna().astype(int)
        
        return df
    
    def create_missingness_features(self, df):
        """Create missingness indicator features"""
        feature_groups = {
            'D_features': ['D1', 'D2', 'D3', 'D4', 'D8', 'D10', 'D15'],
            'card_features': ['card2', 'card3', 'card5', 'card6'],
            'identity_features': [f'id_{i:02d}' for i in range(1, 12)] + 
                               ['id_30', 'id_31', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38'],
            'email_features': ['P_emaildomain', 'R_emaildomain'],
            'M_features': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6'],
            'address_features': ['addr1', 'addr2']
        }
        
        for group_name, features in feature_groups.items():
            available_features = [f for f in features if f in df.columns]
            if available_features:
                df[f'missing_{group_name}'] = df[available_features].isnull().mean(axis=1)
        
        # Overall missingness
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col not in ['TransactionID', 'isFraud']]
        if numerical_cols:
            df['missing_overall'] = df[numerical_cols].isnull().mean(axis=1)
            df['is_high_missing'] = (df['missing_overall'] > 0.5).astype(int)
        
        return df
    
    def create_composite_risk_score(self, df):
        """Create composite fraud risk score"""
        fraud_risk_weights = {
            'is_protonmailP': 10, 'is_protonmailR': 10, 'is_single_use_card': 8,
            'is_high_risk_product': 7, 'is_amount_outlier': 6, 'is_high_risk_OS': 7,
            'is_high_risk_browser': 7, 'C1_outlier': 6, 'C2_outlier': 6,
            'is_fraud_peak_hour': 5, 'is_high_risk_card4': 5, 'is_credit_card': 4,
            'email_match': 3, 'is_weekend': 4
        }
        
        risk_score = 0
        weights_used = 0
        
        for feature, weight in fraud_risk_weights.items():
            if feature in df.columns:
                risk_score += df[feature] * weight
                weights_used += weight
        
        if weights_used > 0:
            df['composite_fraud_risk'] = (risk_score / weights_used) * 100
        else:
            df['composite_fraud_risk'] = 0
        
        return df
    
    def encode_categorical_features(self, df, label_encoders=None, frequency_encoders=None):
        """Encode categorical features"""
        if label_encoders is None:
            label_encoders = {}
        if frequency_encoders is None:
            frequency_encoders = {}
        
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_columns:
            if col not in ['TransactionID', 'isFraud']:
                n_unique = df[col].nunique()
                
                if n_unique <= 10:
                    # Low cardinality - label encoding
                    if col not in label_encoders:
                        le = LabelEncoder()
                        unique_vals = df[col].fillna('unknown').unique()
                        le.fit(unique_vals)
                        label_encoders[col] = le
                    
                    df[f'{col}_encoded'] = label_encoders[col].transform(df[col].fillna('unknown'))
                    
                else:
                    # High cardinality - frequency encoding
                    if col not in frequency_encoders:
                        freq_map = df[col].value_counts().to_dict()
                        frequency_encoders[col] = freq_map
                    
                    df[f'{col}_freq'] = df[col].map(frequency_encoders[col])
                    df[f'{col}_freq'] = df[f'{col}_freq'].fillna(1)
        
        return df, label_encoders, frequency_encoders
    
    def save_engineered_data(self, train_data, test_data):
        """Save fully engineered datasets"""
        train_data.to_parquet(f"{self.DATA_FEATURES_PATH}train_features_fully_engineered.parquet", index=False)
        test_data.to_parquet(f"{self.DATA_FEATURES_PATH}test_features_fully_engineered.parquet", index=False)
        
        print("Saved fully engineered features:")
        print(f"Train: {self.DATA_FEATURES_PATH}train_features_fully_engineered.parquet")
        print(f"Test: {self.DATA_FEATURES_PATH}test_features_fully_engineered.parquet")
        print(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")
    
    def run(self):
        """Execute complete feature engineering pipeline"""
        print("=== STARTING FEATURE ENGINEERING ===")
        
        # Load processed data
        train_data, test_data = self.load_processed_data()
        
        # Strategic feature selection
        train_data_selected, test_data_selected = self.strategic_feature_selection(train_data, test_data)
        
        # Apply feature engineering
        print("Engineering transaction amount features...")
        train_data_engineered = self.create_transaction_amount_features(train_data_selected)
        test_data_engineered = self.create_transaction_amount_features(test_data_selected, train_data_selected)
        
        print("Engineering temporal features...")
        train_data_engineered = self.create_temporal_features(train_data_engineered)
        test_data_engineered = self.create_temporal_features(test_data_engineered)
        
        print("Engineering product features...")
        train_data_engineered = self.create_product_features(train_data_engineered)
        test_data_engineered = self.create_product_features(test_data_engineered)
        
        print("Engineering card features...")
        train_data_engineered = self.create_card_features(train_data_engineered)
        test_data_engineered = self.create_card_features(test_data_engineered, train_data_engineered)
        
        print("Engineering email features...")
        train_data_engineered = self.create_email_features(train_data_engineered, train_data)
        test_data_engineered = self.create_email_features(test_data_engineered, train_data)
        
        print("Engineering identity features...")
        train_data_engineered = self.create_identity_features(train_data_engineered)
        test_data_engineered = self.create_identity_features(test_data_engineered)
        
        print("Engineering missingness features...")
        train_data_engineered = self.create_missingness_features(train_data_engineered)
        test_data_engineered = self.create_missingness_features(test_data_engineered)
        
        print("Creating composite risk score...")
        train_data_engineered = self.create_composite_risk_score(train_data_engineered)
        test_data_engineered = self.create_composite_risk_score(test_data_engineered)
        
        # Encode categorical features
        print("Encoding categorical features...")
        train_data_encoded, label_encoders, frequency_encoders = self.encode_categorical_features(train_data_engineered)
        test_data_encoded, _, _ = self.encode_categorical_features(test_data_engineered, label_encoders, frequency_encoders)
        
        # Drop original categorical columns
        categorical_cols = train_data_encoded.select_dtypes(include=['object']).columns.tolist()
        cols_to_drop = [col for col in categorical_cols if col not in ['TransactionID', 'isFraud']]
        train_data_encoded = train_data_encoded.drop(columns=cols_to_drop)
        test_data_encoded = test_data_encoded.drop(columns=[col for col in cols_to_drop if col in test_data_encoded.columns])
        
        # Handle remaining missing values
        train_data_encoded = train_data_encoded.fillna(-999)
        test_data_encoded = test_data_encoded.fillna(-999)
        
        print(f"After encoding - Train: {train_data_encoded.shape}, Test: {test_data_encoded.shape}")
        
        # Save engineered data
        self.save_engineered_data(train_data_encoded, test_data_encoded)
        
        print("=== FEATURE ENGINEERING COMPLETED SUCCESSFULLY! ===")

if __name__ == "__main__":
    feature_eng = FeatureEngineering()
    feature_eng.run()