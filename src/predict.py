import joblib
import pandas as pd

from src.config import MODEL_PATH


def load_pipeline():
    saved_object = joblib.load(MODEL_PATH)
    return saved_object["pipeline"]


def predict_churn(input_data):
    pipeline = load_pipeline()

    input_df = pd.DataFrame([input_data])

    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    return prediction, probability

