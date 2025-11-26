import pandas as pd
import numpy as np
import joblib
import os

class FraudPredictor:
    def __init__(self, model_path=None):
        self.model_path = model_path or "../models/"
        self.model = None
        self.preprocessing_info = None
        
    def load_model(self, model_name="best_pipeline_xgboost.pkl"):
        """Load trained model and preprocessing info"""
        model_file = os.path.join(self.model_path, model_name)
        info_file = os.path.join(self.model_path, "preprocessing_info.pkl")
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        self.model = joblib.load(model_file)
        self.preprocessing_info = joblib.load(info_file)
        
        print(f"Loaded model: {model_name}")
        print(f"Model features: {len(self.preprocessing_info['features'])}")
        print(f"Validation AUC: {self.preprocessing_info['val_auc']:.4f}")
        
    def preprocess_new_data(self, new_data):
        """Preprocess new data for prediction"""
        # Ensure all required features are present
        required_features = self.preprocessing_info['features']
        missing_features = set(required_features) - set(new_data.columns)
        
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            # Add missing features with default values
            for feature in missing_features:
                new_data[feature] = -999
        
        # Reorder features to match training
        new_data = new_data[required_features]
        
        return new_data
    
    def predict(self, new_data, return_probabilities=True):
        """Generate predictions for new data"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess data
        processed_data = self.preprocess_new_data(new_data)
        
        # Generate predictions
        if return_probabilities:
            predictions = self.model.predict_proba(processed_data)[:, 1]
        else:
            predictions = self.model.predict(processed_data)
        
        return predictions
    
    def predict_batch(self, data_path, output_path=None):
        """Generate predictions for a batch of data"""
        # Load data
        if data_path.endswith('.parquet'):
            data = pd.read_parquet(data_path)
        else:
            data = pd.read_csv(data_path)
        
        # Generate predictions
        predictions = self.predict(data)
        
        # Create results DataFrame
        if 'TransactionID' in data.columns:
            results = pd.DataFrame({
                'TransactionID': data['TransactionID'],
                'isFraud_probability': predictions
            })
        else:
            results = pd.DataFrame({
                'isFraud_probability': predictions
            })
        
        # Save results
        if output_path:
            if output_path.endswith('.parquet'):
                results.to_parquet(output_path, index=False)
            else:
                results.to_csv(output_path, index=False)
            print(f"Predictions saved to: {output_path}")
        
        return results
    
    def evaluate_prediction_quality(self, predictions, threshold=0.5):
        """Evaluate the quality of predictions"""
        fraud_rate = (predictions > threshold).mean() * 100
        print(f"Predicted fraud rate: {fraud_rate:.2f}%")
        print(f"Prediction range: {predictions.min():.4f} - {predictions.max():.4f}")
        
        return fraud_rate

if __name__ == "__main__":
    # Example usage
    predictor = FraudPredictor()
    predictor.load_model()
    
    # Example: Predict on new data
    # new_data = pd.read_parquet("../data/processed/test_features_selected_enhanced.parquet")
    # predictions = predictor.predict_batch("../data/processed/test_features_selected_enhanced.parquet", 
    #                                      "../data/predictions/final_predictions.csv")