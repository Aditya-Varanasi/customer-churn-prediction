import joblib
import pandas as pd

model = joblib.load("models/churn_model.pkl")

# try:
#     model = joblib.load('models/churn_model.pkl')
# except FileNotFoundError:
#     print("Error: Could not find 'models/churn_model.pkl'. Run train.py first.")
#     exit()

sample = pd.DataFrame({
    "gender": ["Male"],
    "SeniorCitizen": [0],   
    "Partner": ["Yes"],           
    "Dependents": ["No"],    
    "tenure": [5],        
    "PhoneService": ["Yes"],   
    "MultipleLines": ["No"],    
    "InternetService": ["Fiber optic"],
    "OnlineSecurity": ["Yes"],
    "OnlineBackup": ["Yes"],
    "DeviceProtection": ["Yes"],
    "TechSupport": ["No"],   
    "StreamingTV": ["No"],       
    "StreamingMovies": ["No"],   
    "Contract": ["Month-to-month"],     
    "PaperlessBilling": ["Yes"],
    "PaymentMethod": ["Mailed check"],     
    "MonthlyCharges": [80.0],
    "TotalCharges": [400.0]
})

# sample_processed['TotalCharges'] = pd.to_numeric(sample_processed['TotalCharges'])

prediction = model.predict(sample)
probability = model.predict_proba(sample)[:, 1]

print(f"Prediction: {prediction[0]}")
print(f"Churn Probability: {probability[0]:.2%}")