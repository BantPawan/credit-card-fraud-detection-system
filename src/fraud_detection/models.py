import os
import time
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit

import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

class ModelSelection:
    def __init__(self):
        self.DATA_FEATURES_PATH = "../data/processed/features/"
        self.MODELS_PATH = "../models/"
        os.makedirs(self.MODELS_PATH, exist_ok=True)
        
    def load_selected_features(self):
        """Load feature-selected data"""
        print("Loading selected features...")
        train_path = f"{self.DATA_FEATURES_PATH}train_features_selected_enhanced.parquet"
        test_path = f"{self.DATA_FEATURES_PATH}test_features_selected_enhanced.parquet"
        
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        
        print(f"Train: {train_df.shape}, Test: {test_df.shape}")
        return train_df, test_df
    
    def prepare_data(self, train_df, test_df):
        """Prepare data for modeling"""
        X = train_df.drop(columns=["TransactionID", "isFraud"])
        y = train_df["isFraud"].astype(int)
        test_ids = test_df["TransactionID"]
        X_test = test_df.drop(columns=["TransactionID"])
        
        # Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        
        print(f"Train rows: {len(X_train)}, Validation rows: {len(X_val)}")
        return X_train, X_val, y_train, y_val, X_test, test_ids
    
    def detect_features(self, df):
        """Detect feature types for preprocessing"""
        nums = df.select_dtypes(include=[np.number]).columns.tolist()
        bin_cols = [c for c in nums if df[c].nunique() == 2]
        cont_cols = [c for c in nums if c not in bin_cols]
        return cont_cols, bin_cols
    
    def stratified_subsample(self, Xd, yd, n, seed=42):
        """Stratified subsample to ensure both classes appear"""
        total = len(Xd)
        if n >= total:
            return Xd.copy(), yd.copy()
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
        idx, _ = next(sss.split(Xd, yd))
        return Xd.iloc[idx], yd.iloc[idx]
    
    def initialize_models(self):
        """Initialize model dictionary with configurations"""
        model_dict = {
            "log_reg": {
                "estimator": LogisticRegression(
                    solver="saga", penalty="l2", C=0.1, class_weight="balanced",
                    max_iter=500, n_jobs=-1, random_state=42
                ),
                "phaseA": {},
                "phaseB": {}
            },
            "random_forest": {
                "estimator": RandomForestClassifier(
                    n_estimators=100, max_depth=12, min_samples_split=100,
                    class_weight="balanced", n_jobs=-1, random_state=42
                ),
                "phaseA": {"n_estimators": 100},
                "phaseB": {"n_estimators": 300}
            },
            "xgboost": {
                "estimator": XGBClassifier(
                    n_estimators=100, eval_metric="auc",
                    scale_pos_weight=10, random_state=42, n_jobs=-1
                ),
                "phaseA": {"n_estimators": 100},
                "phaseB": {"n_estimators": 300}
            },
            "lightgbm": {
                "estimator": LGBMClassifier(
                    n_estimators=100, is_unbalance=True, metric="auc",
                    n_jobs=-1, random_state=42
                ),
                "phaseA": {"n_estimators": 100},
                "phaseB": {"n_estimators": 300}
            }
        }
        return model_dict
    
    def evaluate_model(self, name, entry, pre, Xd, yd, Xv, yv, phase):
        """Evaluate a single model"""
        model = clone(entry["estimator"])
        if phase == "A":
            params = entry["phaseA"]
        else:
            params = entry["phaseB"]
        for k, v in params.items():
            if hasattr(model, k):
                setattr(model, k, v)
        
        X_use, y_use = Xd, yd
        
        pipe = Pipeline([("pre", pre), ("model", model)])
        cv = 3 if phase == "A" else 5
        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Cross-validation
        auc_scores = cross_val_score(pipe, X_use, y_use, cv=kf, scoring="roc_auc", n_jobs=-1)
        ap_scores = cross_val_score(pipe, X_use, y_use, cv=kf, scoring="average_precision", n_jobs=-1)
        
        # Fit and validate
        pipe.fit(X_use, y_use)
        val_p = pipe.predict_proba(Xv)[:, 1]
        val_auc = roc_auc_score(yv, val_p)
        val_ap = average_precision_score(yv, val_p)
        
        return [name, auc_scores.mean(), auc_scores.std(),
                ap_scores.mean(), ap_scores.std(), val_auc, val_ap], pipe
    
    def run_two_phase_evaluation(self, model_dict, X_train, y_train, X_val, y_val):
        """Run two-phase model evaluation"""
        # Detect feature types
        cont_cols, bin_cols = self.detect_features(X_train)
        
        # Create preprocessors
        pre_basic = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), cont_cols),
                ("binary", "passthrough", bin_cols)
            ],
            remainder="passthrough"
        )
        pre_pass = ColumnTransformer(transformers=[], remainder="passthrough")
        
        # Phase A: Quick screening
        resultsA = []
        pipesA = {}
        print("Phase A: Screening...")
        for name, entry in model_dict.items():
            pre = pre_pass if name in {"random_forest", "xgboost", "lightgbm"} else pre_basic
            out, pipe = self.evaluate_model(name, entry, pre, X_train, y_train, X_val, y_val, "A")
            resultsA.append(out)
            pipesA[name] = pipe
            print(f"{name}: {out[1]:.4f} AUC")
        
        dfA = pd.DataFrame(resultsA, 
                          columns=["name", "cv_auc", "cv_std", "cv_ap", "cv_ap_std", "val_auc", "val_ap"])
        dfA = dfA.sort_values("cv_auc", ascending=False)
        
        # Phase B: Detailed evaluation of top models
        top_models = dfA.head(3)["name"].tolist()
        resultsB = []
        pipesB = {}
        print("\nPhase B: Final evaluation...")
        for name in top_models:
            entry = model_dict[name]
            pre = pre_pass if name in {"random_forest", "xgboost", "lightgbm"} else pre_basic
            out, pipe = self.evaluate_model(name, entry, pre, X_train, y_train, X_val, y_val, "B")
            resultsB.append(out)
            pipesB[name] = pipe
            print(f"{name}: {out[1]:.4f} AUC")
        
        dfB = pd.DataFrame(resultsB, 
                          columns=["name", "cv_auc", "cv_std", "cv_ap", "cv_ap_std", "val_auc", "val_ap"])
        dfB = dfB.sort_values("val_auc", ascending=False)
        
        return dfA, dfB, pipesA, pipesB, pre_basic, pre_pass
    
    def hyperparameter_tuning(self, best_model_name, best_pipe, X_train, y_train, preproc):
        """Perform hyperparameter tuning for the best model"""
        print(f"\nRunning hyperparameter tuning for {best_model_name}...")
        
        # Define parameter grids
        param_grid = {}
        if best_model_name == "lightgbm":
            param_grid = {
                'model__n_estimators': [200, 300],
                'model__num_leaves': [31, 63],
                'model__learning_rate': [0.05, 0.1],
                'model__min_child_samples': [20, 50]
            }
        elif best_model_name == "xgboost":
            param_grid = {
                'model__n_estimators': [200, 300],
                'model__max_depth': [6, 8],
                'model__learning_rate': [0.05, 0.1],
                'model__min_child_weight': [1, 5]
            }
        elif best_model_name == "random_forest":
            param_grid = {
                'model__n_estimators': [200, 300],
                'model__max_depth': [12, 15],
                'model__min_samples_leaf': [10, 20]
            }
        
        if param_grid:
            base_pipe = clone(best_pipe)
            grid_search = GridSearchCV(
                base_pipe, param_grid, cv=3, scoring='roc_auc', 
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            print(f"Best params: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_
        else:
            return best_pipe
    
    def evaluate_final_model(self, final_pipe, X_val, y_val):
        """Evaluate final model performance"""
        val_prob = final_pipe.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_prob)
        val_ap = average_precision_score(y_val, val_prob)
        
        print("\nFinal Validation Performance:")
        print(f"AUC: {val_auc:.4f}")
        print(f"Average Precision: {val_ap:.4f}")
        
        # Classification report
        y_pred = (val_prob > 0.5).astype(int)
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"{self.MODELS_PATH}confusion_matrix.png")
        plt.show()
        
        return val_auc, val_ap
    
    def save_results(self, final_pipe, best_model_name, X_train, y_train, X_test, test_ids, val_auc, val_ap):
        """Save model, predictions, and metadata"""
        # Retrain on full training data
        final_pipe.fit(pd.concat([X_train, X_train]), pd.concat([y_train, y_train]))
        
        # Save model
        model_path = os.path.join(self.MODELS_PATH, f"best_pipeline_{best_model_name}.pkl")
        joblib.dump(final_pipe, model_path)
        
        # Generate predictions
        test_pred = final_pipe.predict_proba(X_test)[:, 1]
        submission = pd.DataFrame({
            "TransactionID": test_ids,
            "isFraud": test_pred
        })
        
        # Save predictions
        pred_path = f"{self.DATA_FEATURES_PATH}model_selection_predictions.csv"
        submission.to_csv(pred_path, index=False)
        
        # Save preprocessing info
        cont_cols, bin_cols = self.detect_features(X_train)
        meta_info = {
            "features": X_train.columns.tolist(),
            "continuous": cont_cols,
            "binary": bin_cols,
            "best_model": best_model_name,
            "val_auc": val_auc,
            "val_ap": val_ap
        }
        joblib.dump(meta_info, os.path.join(self.MODELS_PATH, "preprocessing_info.pkl"))
        
        print(f"\nSaved results:")
        print(f"Model: {model_path}")
        print(f"Predictions: {pred_path}")
        print(f"Metadata: {self.MODELS_PATH}preprocessing_info.pkl")
        
        return submission
    
    def run(self):
        """Execute complete model selection pipeline"""
        print("=== STARTING MODEL SELECTION ===")
        start_time = time.time()
        
        # Load data
        train_df, test_df = self.load_selected_features()
        X_train, X_val, y_train, y_val, X_test, test_ids = self.prepare_data(train_df, test_df)
        
        # Initialize models
        model_dict = self.initialize_models()
        
        # Two-phase evaluation
        phaseA_df, phaseB_df, pipesA, pipesB, pre_basic, pre_pass = self.run_two_phase_evaluation(
            model_dict, X_train, y_train, X_val, y_val
        )
        
        print("\nPhase A Results:")
        print(phaseA_df.round(4))
        print("\nPhase B Results:")
        print(phaseB_df.round(4))
        
        # Select best model
        best_model_name = phaseB_df.iloc[0]["name"]
        best_pipe = pipesB[best_model_name]
        print(f"\nBest model: {best_model_name}")
        
        # Hyperparameter tuning
        preproc = pre_pass if best_model_name in {"random_forest", "xgboost", "lightgbm"} else pre_basic
        final_pipe = self.hyperparameter_tuning(best_model_name, best_pipe, X_train, y_train, preproc)
        
        # Final evaluation
        val_auc, val_ap = self.evaluate_final_model(final_pipe, X_val, y_val)
        
        # Save results
        submission = self.save_results(final_pipe, best_model_name, X_train, y_train, X_test, test_ids, val_auc, val_ap)
        
        execution_time = time.time() - start_time
        print(f"\n=== MODEL SELECTION COMPLETED SUCCESSFULLY! ===")
        print(f"Total execution time: {execution_time:.2f} seconds")
        print(f"Best model: {best_model_name} with validation AUC: {val_auc:.4f}")

if __name__ == "__main__":
    model_sel = ModelSelection()
    model_sel.run()