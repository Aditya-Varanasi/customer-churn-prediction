from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("models/churn_model.pkl")

@app.get("/")
def home():
    return {"message": "Churn prediction API"}

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return{
        "prediction": int(prediction),
        "churn_probability": float(probability) 
    }
