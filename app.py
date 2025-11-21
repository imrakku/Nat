# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

st.set_page_config(page_title="Multi-Product Spend + Cluster Predictor", layout="centered")
st.title("🛒 Multi-Product Spend & Cluster Predictor")

# -----------------------
# Files expected
# -----------------------
FILES = {
    "models": "all_models.pkl",
    "features": "feature_columns.json",
    "defaults": "default_values.json",
    "kmeans": "kmeans_model.pkl",
    "scaler": "cluster_scaler.pkl"
}

# -----------------------
# Load resources (safe)
# -----------------------
missing = []
for k,f in FILES.items():
    if not os.path.exists(f):
        missing.append(f)

if missing:
    st.error(f"The following required files are missing: {', '.join(missing)}")
    st.info("Make sure all files are in the app folder: all_models.pkl, feature_columns.json, default_values.json, kmeans_model.pkl, cluster_scaler.pkl")
    st.stop()

models = joblib.load(FILES["models"])            # dict: model name -> sklearn model
feature_cols = json.load(open(FILES["features"])) 
defaults = json.load(open(FILES["defaults"]))
kmeans = joblib.load(FILES["kmeans"])
scaler = joblib.load(FILES["scaler"])

# -----------------------
# Segment names mapping (as confirmed)
# -----------------------
segment_names = {
    0: "Budget-Conscious Light Shoppers",
    1: "Value-Driven Mid-Spend Families",
    2: "Educated Low-Spend Traditionalists",
    3: "Elite High-End Gourmet Spenders",
    4: "Affluent Multi-Category Power Shoppers",
    5: "Premium Balanced Lifestyle Shoppers"
}

# -----------------------
# Cluster profile (means) — embedded from the provided table
# Replace / update these numbers if you re-run profiling and want to persist
# -----------------------
cluster_profile_data = {
    0: {'Kidhome':0.8,'Teenhome':0.5,'Recency':48.5,'MntWines':43.5,'MntFruits':4.7,'MntMeatProducts':22.1,'MntFishProducts':6.4,'MntSweetProducts':4.6,'MntGoldProds':15.3,'NumDealsPurchases':2.0,'NumWebPurchases':None,'NumCatalogPurchases':None,'NumStorePurchases':None,'NumWebVisitsMonth':None,'AcceptedCmp3':None,'AcceptedCmp4':None,'AcceptedCmp5':None,'AcceptedCmp1':None,'AcceptedCmp2':None,'Complain':None,'Response':None,'Age':None,'Marital_Status_Alone':None,'Marital_Status_Divorced':None,'Marital_Status_Married':None,'Marital_Status_Single':None,'Marital_Status_Together':None,'Marital_Status_Widow':None,'Marital_Status_YOLO':None,'Education_Basic':0.1,'Education_Graduation':0.0,'Education_Master':0.3,'Education_PhD':0.4,'Income_imputed':35819.8},
    1: {'Kidhome':0.3,'Teenhome':1.0,'Recency':46.6,'MntWines':472.5,'MntFruits':15.5,'MntMeatProducts':117.1,'MntFishProducts':20.5,'MntSweetProducts':16.5,'MntGoldProds':59.9,'NumDealsPurchases':4.3,'NumWebPurchases':None,'NumCatalogPurchases':None,'NumStorePurchases':None,'NumWebVisitsMonth':None,'AcceptedCmp3':None,'AcceptedCmp4':None,'AcceptedCmp5':None,'AcceptedCmp1':None,'AcceptedCmp2':None,'Complain':None,'Response':None,'Age':None,'Marital_Status_Alone':None,'Marital_Status_Divorced':None,'Marital_Status_Married':None,'Marital_Status_Single':None,'Marital_Status_Together':None,'Marital_Status_Widow':None,'Marital_Status_YOLO':None,'Education_Basic':0.0,'Education_Graduation':0.4,'Education_Master':0.2,'Education_PhD':0.3,'Income_imputed':56238.0},
    2: {'Kidhome':0.8,'Teenhome':0.4,'Recency':50.9,'MntWines':46.9,'MntFruits':6.1,'MntMeatProducts':28.2,'MntFishProducts':9.3,'MntSweetProducts':6.1,'MntGoldProds':18.0,'NumDealsPurchases':2.2,'NumWebPurchases':None,'NumCatalogPurchases':None,'NumStorePurchases':None,'NumWebVisitsMonth':None,'AcceptedCmp3':None,'AcceptedCmp4':None,'AcceptedCmp5':None,'AcceptedCmp1':None,'AcceptedCmp2':None,'Complain':None,'Response':None,'Age':None,'Marital_Status_Alone':None,'Marital_Status_Divorced':None,'Marital_Status_Married':None,'Marital_Status_Single':None,'Marital_Status_Together':None,'Marital_Status_Widow':None,'Marital_Status_YOLO':None,'Education_Basic':0.0,'Education_Graduation':1.0,'Education_Master':0.0,'Education_PhD':0.0,'Income_imputed':35018.0},
    3: {'Kidhome':0.1,'Teenhome':0.1,'Recency':49.7,'MntWines':899.4,'MntFruits':50.9,'MntMeatProducts':463.5,'MntFishProducts':72.4,'MntSweetProducts':60.3,'MntGoldProds':72.9,'NumDealsPurchases':1.0,'NumWebPurchases':None,'NumCatalogPurchases':None,'NumStorePurchases':None,'NumWebVisitsMonth':None,'AcceptedCmp3':None,'AcceptedCmp4':None,'AcceptedCmp5':None,'AcceptedCmp1':None,'AcceptedCmp2':None,'Complain':None,'Response':None,'Age':None,'Marital_Status_Alone':None,'Marital_Status_Divorced':None,'Marital_Status_Married':None,'Marital_Status_Single':None,'Marital_Status_Together':None,'Marital_Status_Widow':None,'Marital_Status_YOLO':None,'Education_Basic':0.0,'Education_Graduation':0.5,'Education_Master':0.2,'Education_PhD':0.3,'Income_imputed':81858.0},
    4: {'Kidhome':0.0,'Teenhome':0.3,'Recency':49.3,'MntWines':478.6,'MntFruits':103.8,'MntMeatProducts':440.2,'MntFishProducts':143.2,'MntSweetProducts':95.5,'MntGoldProds':104.5,'NumDealsPurchases':1.6,'NumWebPurchases':None,'NumCatalogPurchases':None,'NumStorePurchases':None,'NumWebVisitsMonth':None,'AcceptedCmp3':None,'AcceptedCmp4':None,'AcceptedCmp5':None,'AcceptedCmp1':None,'AcceptedCmp2':None,'Complain':None,'Response':None,'Age':None,'Marital_Status_Alone':None,'Marital_Status_Divorced':None,'Marital_Status_Married':None,'Marital_Status_Single':None,'Marital_Status_Together':None,'Marital_Status_Widow':None,'Marital_Status_YOLO':None,'Education_Basic':0.0,'Education_Graduation':0.8,'Education_Master':0.1,'Education_PhD':0.1,'Income_imputed':72478.0},
    5: {'Kidhome':0.1,'Teenhome':0.4,'Recency':50.0,'MntWines':499.6,'MntFruits':45.4,'MntMeatProducts':353.1,'MntFishProducts':69.2,'MntSweetProducts':49.6,'MntGoldProds':58.7,'NumDealsPurchases':1.7,'NumWebPurchases':None,'NumCatalogPurchases':None,'NumStorePurchases':None,'NumWebVisitsMonth':None,'AcceptedCmp3':None,'AcceptedCmp4':None,'AcceptedCmp5':None,'AcceptedCmp1':None,'AcceptedCmp2':None,'Complain':None,'Response':None,'Age':None,'Marital_Status_Alone':None,'Marital_Status_Divorced':None,'Marital_Status_Married':None,'Marital_Status_Single':None,'Marital_Status_Together':None,'Marital_Status_Widow':None,'Marital_Status_YOLO':None,'Education_Basic':0.0,'Education_Graduation':0.5,'Education_Master':0.2,'Education_PhD':0.3,'Income_imputed':72050.8}
}
cluster_profile = pd.DataFrame(cluster_profile_data).T

# -----------------------
# UI - minimal inputs
# -----------------------
st.header("Enter customer partial info (only these are required)")

col1, col2 = st.columns(2)
with col1:
    income = st.number_input("Income", value=float(defaults.get("Income", 50000.0)))
    age = st.number_input("Age", value=float(defaults.get("Age", 40.0)), min_value=16, max_value=100)
with col2:
    kids = st.number_input("Kidhome", min_value=0, max_value=10, value=int(defaults.get("Kidhome", 0)))
    teens = st.number_input("Teenhome", min_value=0, max_value=10, value=int(defaults.get("Teenhome", 0)))

marital = st.selectbox("Marital Status", ["Alone","Divorced","Married","Single","Together","Widow","YOLO"])

st.markdown("Click **Predict** to get six spend predictions + cluster assignment + cluster profile.")

if st.button("Predict"):
    # -----------------------
    # Build full input row
    # -----------------------
    row = {}
    # One-hot marital columns detection
    marital_cols = [c for c in feature_cols if c.startswith("Marital_Status_")]

    for c in feature_cols:
        if c == "Income" or c.lower() == "income_imputed":
            row[c] = float(income)
        elif c == "Age":
            row[c] = float(age)
        elif c == "Kidhome":
            row[c] = int(kids)
        elif c == "Teenhome":
            row[c] = int(teens)
        elif c in marital_cols:
            row[c] = 1 if c == f"Marital_Status_{marital}" else 0
        else:
            # fallback to defaults (mean/mode)
            # defaults may be strings for categorical; coerce numeric cols
            val = defaults.get(c, 0)
            try:
                row[c] = float(val)
            except:
                row[c] = val

    input_df = pd.DataFrame([row])[feature_cols]  # ensure correct order

    st.subheader("Full input used by models")
    st.dataframe(input_df.T, width=700)

    # -----------------------
    # Predict all models
    # -----------------------
    preds = {}
    for name, m in models.items():
        try:
            p = m.predict(input_df)[0]
            preds[name] = float(np.round(p,2))
        except Exception as e:
            preds[name] = f"Error: {e}"

    # display predictions
    st.subheader("Predicted Spend (per product)")
    pred_df = pd.DataFrame.from_dict(preds, orient='index', columns=['Predicted'])
    st.table(pred_df)

    # -----------------------
    # Cluster assignment for the single row
    # -----------------------
    # Use the same scaler used in training clustering
    # some scalers expect the features in the same order as during training; scaler was fit on FEATURE_COLS
    to_scale = input_df.copy()
    # If scaler was fit on a different set (e.g., df_clust), ensure to pick same columns — here we assume they match
    try:
        Xs = scaler.transform(to_scale)
    except Exception as e:
        st.warning(f"Scaler transform failed: {e}. Attempting to align columns.")
        # try to align by columns intersection and fill missing with 0
        cols_needed = scaler.mean_.shape[0]
        to_scale_fixed = to_scale.reindex(columns=feature_cols, fill_value=0)
        Xs = scaler.transform(to_scale_fixed)

    cluster_label = int(kmeans.predict(Xs)[0])
    seg_name = segment_names.get(cluster_label, f"Segment {cluster_label}")

    st.subheader("Cluster Assignment")
    st.write(f"**Cluster:** {cluster_label}")
    st.write(f"**Segment Name:** {seg_name}")

    # -----------------------
    # Show cluster profile (the mean row)
    # -----------------------
    if cluster_label in cluster_profile.index:
        st.subheader("Cluster Profile (means)")
        display_cols = ['MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds','Income_imputed','Kidhome','Teenhome','Recency','NumDealsPurchases']
        # pick available columns from profile
        display_cols_present = [c for c in display_cols if c in cluster_profile.columns]
        prof_row = cluster_profile.loc[cluster_label, display_cols_present].to_frame(name="Mean").T
        st.table(prof_row)
    else:
        st.info("No saved profile available for this cluster.")

    # -----------------------
    # Option: download input + preds
    # -----------------------
    result_out = input_df.copy()
    for k,v in preds.items():
        result_out[k + "_pred"] = v
    result_out["Cluster"] = cluster_label
    csv = result_out.to_csv(index=False).encode('utf-8')
    st.download_button("Download prediction row (CSV)", data=csv, file_name="prediction_row.csv", mime="text/csv")
