import streamlit as st
import pandas as pd
import joblib

from src.config import DATA_PATH, MODEL_PATH
from src.predict import predict_churn


st.set_page_config(
    page_title="Customer Churn Intelligence Platform",
    layout="wide"
)


@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    return df


@st.cache_resource
def load_saved_model():
    return joblib.load(MODEL_PATH)


def get_risk_level(probability):
    if probability >= 0.7:
        return "High Risk", "red"
    elif probability >= 0.4:
        return "Medium Risk", "orange"
    else:
        return "Low Risk", "green"


def generate_recommendations(probability, input_data):
    recommendations = []

    if probability >= 0.7:
        recommendations.append("Prioritise this customer for retention follow-up.")
        recommendations.append("Consider offering a loyalty discount or service bundle.")
        recommendations.append("Review service quality issues before the customer switches provider.")
    elif probability >= 0.4:
        recommendations.append("Monitor this customer and consider proactive engagement.")
        recommendations.append("Offer contract upgrade options or personalised support.")
    else:
        recommendations.append("Customer appears stable, but continue maintaining service quality.")

    if input_data["Contract"] == "Month-to-month":
        recommendations.append("Encourage transition from month-to-month to a longer-term contract.")

    if input_data["TechSupport"] == "No":
        recommendations.append("Promote technical support services to improve customer confidence.")

    if input_data["PaymentMethod"] == "Electronic check":
        recommendations.append("Suggest automatic payment methods to improve retention stability.")

    return recommendations


def generate_explanations(input_data):
    explanations = []

    if input_data["Contract"] == "Month-to-month":
        explanations.append("Month-to-month contract customers usually have higher churn risk.")

    if input_data["tenure"] <= 12:
        explanations.append("Short customer tenure indicates the customer is still in the early stage of the relationship.")

    if input_data["MonthlyCharges"] >= 75:
        explanations.append("High monthly charges may increase price sensitivity.")

    if input_data["TechSupport"] == "No":
        explanations.append("Lack of technical support may reduce customer satisfaction.")

    if input_data["InternetService"] == "Fiber optic":
        explanations.append("Fiber optic customers in this dataset show higher churn patterns.")

    if input_data["PaymentMethod"] == "Electronic check":
        explanations.append("Electronic check payment is associated with higher churn behaviour in the dataset.")

    if not explanations:
        explanations.append("No major high-risk pattern was detected from the selected customer details.")

    return explanations


df = load_dataset()
saved_object = load_saved_model()

st.title("Customer Churn Intelligence Platform")
st.caption("AI-powered customer retention decision support system")

st.write(
    "This project uses machine learning to predict customer churn risk and present "
    "business-friendly insights that can support customer retention decisions."
)

st.divider()

# Executive dashboard
st.header("Executive Dashboard")

total_customers = len(df)
churned_customers = df[df["Churn"] == "Yes"].shape[0]
churn_rate = churned_customers / total_customers * 100
avg_monthly_charges = df["MonthlyCharges"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Churned Customers", f"{churned_customers:,}")
col3.metric("Churn Rate", f"{churn_rate:.2f}%")
col4.metric("Avg Monthly Charges", f"${avg_monthly_charges:.2f}")

st.divider()

# Business analytics
st.header("Business Analytics")

left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Churn Rate by Contract Type")

    contract_churn = pd.crosstab(
        df["Contract"],
        df["Churn"],
        normalize="index"
    ) * 100

    st.bar_chart(contract_churn["Yes"])

    st.info(
        "Customers on month-to-month contracts have the highest churn rate. "
        "Longer contracts are strongly associated with better retention."
    )

with right_col:
    st.subheader("Churn Rate by Internet Service")

    internet_churn = pd.crosstab(
        df["InternetService"],
        df["Churn"],
        normalize="index"
    ) * 100

    st.bar_chart(internet_churn["Yes"])

    st.info(
        "Internet service type also shows different churn behaviour. "
        "This can help identify service groups that may need targeted support."
    )

st.subheader("Monthly Charges by Churn Status")

monthly_summary = df.groupby("Churn")["MonthlyCharges"].mean()
st.bar_chart(monthly_summary)

st.info(
    "Customers who churn tend to have higher average monthly charges, which may indicate price sensitivity."
)

st.divider()

# Prediction section
st.header("Customer Risk Assessment")

st.write(
    "Enter customer information below to generate a churn risk prediction and retention recommendation."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Customer Profile")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure in Months", 0, 72, 5)

with col2:
    st.subheader("Service Details")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

with col3:
    st.subheader("Billing Details")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=90.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=450.0)

input_data = {
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

if st.button("Generate Risk Assessment"):
    prediction, probability = predict_churn(input_data)
    risk_level, risk_colour = get_risk_level(probability)

    st.subheader("Prediction Result")

    if risk_level == "High Risk":
        st.error(f"{risk_level}: {probability:.2%} probability of churn")
    elif risk_level == "Medium Risk":
        st.warning(f"{risk_level}: {probability:.2%} probability of churn")
    else:
        st.success(f"{risk_level}: {probability:.2%} probability of churn")

    st.progress(float(probability))

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.subheader("Recommended Actions")
        for item in generate_recommendations(probability, input_data):
            st.write(f"- {item}")

    with result_col2:
        st.subheader("Why This Customer May Be At Risk")
        for item in generate_explanations(input_data):
            st.write(f"- {item}")

st.divider()

# Model insights
st.header("Model Explainability")

pipeline = saved_object["pipeline"]
model = pipeline.named_steps["model"]
preprocessor = pipeline.named_steps["preprocessor"]

cat_features = saved_object["categorical_features"]
num_features = saved_object["numerical_features"]

encoded_cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_features)
all_features = list(encoded_cat_features) + num_features

feature_importance = pd.DataFrame({
    "Feature": all_features,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.subheader("Top Features Influencing Churn Prediction")
st.bar_chart(feature_importance.head(10).set_index("Feature")["Importance"])

with st.expander("View feature importance table"):
    st.dataframe(feature_importance.head(15))

st.info(
    "Feature importance shows which inputs the Random Forest model used most strongly "
    "when making churn predictions. This improves transparency and helps connect model "
    "outputs to business factors."
)

st.divider()

# About section
st.header("About This Project")

st.write(
    """
    This project demonstrates an end-to-end machine learning workflow for customer churn prediction.
    It includes data cleaning, exploratory data analysis, preprocessing with a Scikit-learn pipeline,
    Random Forest model training, model evaluation, churn probability prediction, and an interactive
    Streamlit dashboard.
    """
)

st.write(
    """
    The model is designed as a decision-support tool rather than a final automated decision system.
    In a real business environment, predictions should be combined with customer context, business rules,
    and human judgement before taking retention action.
    """
)