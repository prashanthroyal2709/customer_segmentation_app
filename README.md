# Customer Segmentation App using Machine Learning

This project is a **Customer Segmentation Web App** built using **Machine Learning and Streamlit**.  
It predicts the type of customer based on their behavior and demographic details.

## 🚀 Live Demo
👉 https://customersegmentation27091.streamlit.app/

## 📌 Problem Statement
Businesses need to understand customer behavior to:
- Identify high-value customers
- Improve targeted marketing
- Increase revenue and retention

This app classifies customers into:
- 💰 Budget Customers
- 🧾 Regular Customers
- 👑 Premium Customers

## 🧠 Machine Learning Approach
- Data preprocessing (scaling & feature selection)
- Dimensionality reduction using **PCA**
- Clustering using **K-Means**
- Cluster labeling based on feature importance

## 🧾 Input Features
- Income
- Recency (days)
- Web Visits per Month
- Age
- Children
- Products Purchased
- Total Purchases
- Accepted Campaign
- Response
- Complain
- Customer Since (months)

## 🎯 Output
The app predicts the **Customer Segment**:
- Budget Customer
- Regular Customer
- Premium Customer

## 🛠 Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- GitHub
- Streamlit Cloud

## ▶️ How to Run Locally
```bash
git clone https://github.com/prashanthroyal2709/REPO_NAME.git
cd REPO_NAME
pip install -r requirements.txt
streamlit run app.py
