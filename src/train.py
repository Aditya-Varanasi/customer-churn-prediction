import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from preprocess import load_data,create_preprocessor

df = load_data("data/Telco_customer_churn.csv")

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

preprocessor = create_preprocessor(df)

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)

print("The accuracy of the model is :", accuracy_score(y_test, pred))

joblib.dump(pipeline, "models/churn_model.pkl")


