import pandas as pd


def load_data(data_path):
    return pd.read_csv(data_path)


def clean_data(df):
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    return df
