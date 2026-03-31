import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

def train_pipeline():
    print("--- Starting Training Pipeline ---")
    
    # load data
    data_path =  "data/creditcard.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not exist")
        return 
    
    df = pd.read_csv(data_path)
    
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    X = df.drop(['Class', 'Time'], axis=1)
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print("Training Best Model...")
    best_model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=None, 
        random_state=42, 
        n_jobs=-1
    )
    best_model.fit(X_train_res, y_train_res)
    
    # Evaluate
    y_pred = best_model.predict(X_test)
    print("\nModel Evaluation on Test Set:")
    print(classification_report(y_test, y_pred))
    
    # Save Objects
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(best_model, 'models/best_model.joblib')
    print("\n Save")

if __name__ == "__main__":
    train_pipeline()