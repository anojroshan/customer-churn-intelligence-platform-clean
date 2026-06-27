import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.config import MODEL_PATH


def evaluate_model():
    saved_object = joblib.load(MODEL_PATH)

    model = saved_object["model"]
    X_test = saved_object["X_test"]
    y_test = saved_object["y_test"]

    y_pred = model.predict(X_test)

    print("Model Evaluation Results")
    print("-" * 30)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    feature_importance = pd.DataFrame({
        "Feature": saved_object["features"],
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))


if __name__ == "__main__":
    evaluate_model()