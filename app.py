
import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(page_title="Multi-Product Spend Prediction")

st.title("🛒 Customer Spend Prediction (All Models)")

# Load models
models = joblib.load("all_models.pkl")

# Load feature list + defaults
feature_cols = json.load(open("feature_columns.json"))
defaults = json.load(open("default_values.json"))

# Inputs required from user
st.header("Input Customer Information")

income = st.number_input("Income", value=50000.0)
age = st.number_input("Age", value=40.0)
kids = st.number_input("Kidhome", value=0)
teens = st.number_input("Teenhome", value=0)

marital = st.selectbox(
    "Marital Status",
    ["Alone", "Divorced", "Married", "Single", "Together", "Widow", "YOLO"]
)

# Build one-hot block for marital
marital_cols = [c for c in feature_cols if c.startswith("Marital_Status_")]

marital_map = {col: 0 for col in marital_cols}
marital_map[f"Marital_Status_{marital}"] = 1

# Build input row
row = {}

for col in feature_cols:
    if col == "Income":
        row[col] = income
    elif col == "Age":
        row[col] = age
    elif col == "Kidhome":
        row[col] = kids
    elif col == "Teenhome":
        row[col] = teens
    elif col in marital_map:
        row[col] = marital_map[col]
    else:
        row[col] = defaults[col]  # mean/mode fallback

input_df = pd.DataFrame([row])

st.subheader("Full Model Input (Auto-Filled)")
st.dataframe(input_df)

if st.button("Predict All"):
    results = {}
    for name, model in models.items():
        pred = model.predict(input_df)[0]
        results[name] = round(float(pred), 2)

    st.header("Results")
    for k, v in results.items():
        st.write(f"**{k}:** {v}")
