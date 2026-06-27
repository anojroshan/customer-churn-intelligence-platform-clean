import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from src.config import DATA_PATH, MODEL_PATH, RANDOM_STATE, TEST_SIZE
from src.preprocess import load_data, clean_data


def train_model():
    df = load_data(DATA_PATH)
    df = clean_data(df)

    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    categorical_features = X.select_dtypes(
        include=["object", "str"]
    ).columns.tolist()
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numerical_features)
        ]
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    pipeline.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        {
            "pipeline": pipeline,
            "X_test": X_test,
            "y_test": y_test,
            "categorical_features": categorical_features,
            "numerical_features": numerical_features
        },
        MODEL_PATH
    )

    print("Pipeline training completed.")
    print(f"Pipeline saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()