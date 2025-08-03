import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load saved models and data
rfm_model = joblib.load("rfm_kmeans_model.pkl")
scaler = joblib.load("rfm_scaler.pkl")
product_similarity = pd.read_pickle("product_similarity.pkl")

st.title("Shopper Spectrum")

tab1, tab2 = st.tabs(["Product Recommender", "Customer Segmentation"])

with tab1:
    st.header("Product Recommendation")
    product_input = st.text_input("Enter Product Code (e.g., 85123A):")
    if st.button("Get Recommendations"):
        if product_input in product_similarity:
            similar_items = product_similarity[product_input].sort_values(ascending=False)[1:6].index.tolist()
            st.success("Top 5 Similar Products:")
            for i, item in enumerate(similar_items, 1):
                st.markdown(f"**{i}.** {item}")
        else:
            st.error("Product not found!")

with tab2:
    st.header(" Customer Segmentation")
    rec = st.number_input("Recency (days):", min_value=0)
    freq = st.number_input("Frequency (purchases):", min_value=0)
    mon = st.number_input("Monetary (total spend):", min_value=0.0)

    if st.button("Predict Cluster"):
        rfm_input = scaler.transform([[rec, freq, mon]])
        cluster = rfm_model.predict(rfm_input)[0]
        label = ["High-Value", "Regular", "Occasional", "At-Risk"][cluster]  # Adjust based on your mapping
        st.success(f"Predicted Segment: **{label}**")
