import joblib
import numpy as np
import pandas as pd

class FraudPredictor:
    def __init__(self, model_path='models/best_model.joblib', scaler_path='models/scaler.joblib'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        self.feature_columns = [f'V{i}' for i in range(1, 29)] + ['Amount']
        
    def preprocess(self, transaction_dict):
        df_input = pd.DataFrame([transaction_dict])
        
        df_input['Amount'] = self.scaler.transform(df_input[['Amount']])
        
        return df_input[self.feature_columns]
    
    def predict(self, transaction_dict):
        processed_data = self.preprocess(transaction_dict)
        
        prediction = int(self.model.predict(processed_data)[0])
        probability = float(self.model.predict_proba(processed_data)[0][1])
        
        return {
            "is_fraud": bool(prediction),
            "fraud_probability": round(probability, 4),
            "prediction_label": "Fraud" if prediction == 1 else "Normal"
        }
        
if __name__ == "__main__":
    test_transaction = {f'V{i}': np.random.randn() for i in range(1, 29)}
    test_transaction['Amount'] = 55.0  
    
    predictor = FraudPredictor()
    result = predictor.predict(test_transaction)
    
    print("--- RESULTS ---")
    print(result)