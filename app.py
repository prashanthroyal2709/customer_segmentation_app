import streamlit as st
import pandas as pd
import joblib
import os

# ==============================
# Load saved models
# ==============================
BASE_DIR = os.path.dirname(__file__)  # folder where app.py is

kmeans = joblib.load(os.path.join(BASE_DIR, "kmeans_customer_segmentation.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
pca = joblib.load(os.path.join(BASE_DIR, "pca_model.pkl"))
final_columns = joblib.load(os.path.join(BASE_DIR, "final_df_columns.pkl"))
numeric_cols = [
    'Income','Recency','NumWebVisitsMonth','Complain','Response',
    'Age','Children','Products','Total_purchases','Acceptedcmp','since'
]

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="Customer Segmentation App", layout="centered")

st.title("🧠 Customer Segmentation Web App")
st.write("Predict customer segment using Machine Learning")

st.divider()

# ==============================
# User Inputs
# ==============================
Income = st.number_input("Income", min_value=0, value=50000)
Recency = st.number_input("Recency (days)", min_value=0, value=10)
NumWebVisitsMonth = st.number_input("Web Visits per Month", min_value=0, value=5)
Complain = st.selectbox("Complain", [0, 1])
Response = st.selectbox("Response", [0, 1])
Age = st.number_input("Age", min_value=18, value=35)
Children = st.number_input("Children", min_value=0, value=1)
Products = st.number_input("Products Purchased", min_value=0, value=3)
Total_purchases = st.number_input("Total Purchases", min_value=0, value=2000)
Acceptedcmp = st.selectbox("Accepted Campaign", [0, 1])
since = st.number_input("Customer Since (months)", min_value=0, value=24)

# ==============================
# Prediction Button
# ==============================
if st.button("Predict Customer Segment"):
    new_customer = {
        'Income': Income,
        'Recency': Recency,
        'NumWebVisitsMonth': NumWebVisitsMonth,
        'Complain': Complain,
        'Response': Response,
        'Age': Age,
        'Children': Children,
        'Products': Products,
        'Total_purchases': Total_purchases,
        'Acceptedcmp': Acceptedcmp,
        'since': since
    }

    df = pd.DataFrame([new_customer])

    # Add missing columns
    for col in final_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns
    df = df[final_columns]

    # Scale numeric columns
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Apply PCA
    df_pca = pca.transform(df[numeric_cols].values)

    # Predict cluster
    cluster = kmeans.predict(df_pca)[0]

    # ==============================
    # Cluster Interpretation
    # ==============================
    if cluster == 0:
        segment = "💎 Premium Customer"
    elif cluster == 1:
        segment = "💰 Budget Customer"
    else:
        segment = "🛍️ Moderate Customer"

    st.success(f"Predicted Segment: {segment}")
