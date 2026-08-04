"""Streamlit front end for the SuperKart sales forecasting API."""

import os

import pandas as pd
import requests
import streamlit as st

# Resolved by Docker Compose to the backend service on the shared network.
BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:7860")
SINGLE_ENDPOINT = f"{BACKEND_URL}/v1/sales/single"
BATCH_ENDPOINT = f"{BACKEND_URL}/v1/sales/batch"

st.set_page_config(page_title="SuperKart Sales Forecast", layout="centered")
st.title("SuperKart Sales Forecast")
st.caption("Forecast total revenue for a product in a store.")

with st.sidebar:
    st.subheader("Service status")
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if health.status_code == 200:
            body = health.json()
            st.success(f"Backend reachable. Model: {body.get('model_name', 'unknown')}")
        else:
            st.warning(f"Backend returned {health.status_code}")
    except requests.RequestException as exc:
        st.error(f"Backend unreachable: {exc}")

st.subheader("Single product forecast")

# Numeric ranges match the observed training data. Tree ensembles cannot
# extrapolate, so inputs are constrained to the region the model has seen.
product_weight = st.number_input(
    "Product weight", min_value=4.0, max_value=22.0, value=12.65, step=0.01, format="%.2f"
)
product_allocated_area = st.number_input(
    "Product allocated area (share of total display area)",
    min_value=0.004, max_value=0.298, value=0.068, step=0.001, format="%.3f",
)
product_mrp = st.number_input(
    "Product MRP (maximum retail price)",
    min_value=31.0, max_value=266.0, value=147.03, step=0.01, format="%.2f",
)
store_age_years = st.number_input(
    "Store age (years)", min_value=0, max_value=60, value=17, step=1
)

product_sugar_content = st.selectbox(
    "Product sugar content", ["Low Sugar", "Regular", "No Sugar"], index=0
)
product_id_char = st.selectbox(
    "Product family code",
    ["FD", "DR", "NC"],
    index=0,
    help="FD = Food, DR = Drinks, NC = Non-Consumable. Taken from the first two characters of Product_Id.",
)
product_type_category = st.selectbox(
    "Product shelf-life category",
    ["Perishables", "Non Perishables"],
    index=0,
    help="Perishables covers dairy, meat, fresh produce, bread, breakfast, seafood and frozen foods.",
)
store_size = st.selectbox("Store size", ["Small", "Medium", "High"], index=1)
store_location_city_type = st.selectbox(
    "Store city tier", ["Tier 1", "Tier 2", "Tier 3"], index=1
)
store_type = st.selectbox(
    "Store type",
    ["Departmental Store", "Food Mart", "Supermarket Type1", "Supermarket Type2"],
    index=3,
)

product_data = {
    "Product_Weight": product_weight,
    "Product_Sugar_Content": product_sugar_content,
    "Product_Allocated_Area": product_allocated_area,
    "Product_MRP": product_mrp,
    "Store_Size": store_size,
    "Store_Location_City_Type": store_location_city_type,
    "Store_Type": store_type,
    "Product_Id_char": product_id_char,
    "Product_Type_Category": product_type_category,
    "Store_Age_Years": store_age_years,
}

if st.button("Predict sales", type="primary"):
    try:
        response = requests.post(SINGLE_ENDPOINT, json=product_data, timeout=30)
    except requests.RequestException as exc:
        st.error(f"Could not reach the API: {exc}")
    else:
        if response.status_code == 200:
            st.success(f"Predicted total sales: {response.json()['Predicted_Sales']:,.2f}")
        else:
            st.error(f"API returned {response.status_code}")
            st.json(response.json())

st.divider()
st.subheader("Batch forecast")
st.caption(
    "Upload a CSV containing the ten feature columns. An optional Product_Id column "
    "is echoed back with each prediction."
)

uploaded = st.file_uploader("CSV file", type=["csv"])
if uploaded is not None:
    preview = pd.read_csv(uploaded)
    st.write(f"{len(preview):,} rows uploaded. First five:")
    st.dataframe(preview.head())
    uploaded.seek(0)

    if st.button("Predict batch sales", type="primary"):
        try:
            response = requests.post(BATCH_ENDPOINT, files={"file": uploaded}, timeout=120)
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")
        else:
            if response.status_code == 200:
                body = response.json()
                results = pd.DataFrame(body["predictions"])
                results.columns = ["Product / row id", "Predicted sales"]
                st.success(f"{body['count']:,} predictions returned.")
                st.dataframe(results, use_container_width=True)
                st.download_button(
                    "Download predictions as CSV",
                    results.to_csv(index=False).encode("utf-8"),
                    file_name="superkart_predictions.csv",
                    mime="text/csv",
                )
            else:
                st.error(f"API returned {response.status_code}")
                st.json(response.json())
