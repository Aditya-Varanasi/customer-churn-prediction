import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def load_data(path):
    df = pd.read_csv(path)

    df.drop(columns=["customerID"],inplace=True)
    df["Churn"] = df["Churn"].map({"No": 0,"Yes": 1})

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df

def create_preprocessor(df):
    X = df.drop("Churn", axis=1)

    num_cols = X.select_dtypes(include=["int64","float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("scaler", OneHotEncoder())
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    return preprocessor