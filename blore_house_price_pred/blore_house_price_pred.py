import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack

# -------------------------------
# Load models and preprocessing
# -------------------------------
# These are your trained models
#lr = joblib.load("linear_regression_1.pkl")
#dtr = joblib.load("decision_tree.pkl")
rfr = joblib.load("random_forest.pkl")
#xr = joblib.load("xgboost_1.pkl")

# Preprocessing objects
imputer = joblib.load("imputer.pkl")          # SimpleImputer for numeric features
scaler = joblib.load("scaler.pkl")            # StandardScaler for numeric features
encoder = joblib.load("onehot_encoder_1.pkl")   # OneHotEncoder for 'location_cleaned'

models = {
    #"Linear Regression": lr,
    #"Decision Tree": dtr,
    "Random Forest": rfr,
    #"XGBoost": xr
}

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🏠 Bengaluru House Price Prediction")
st.write("Enter house details to predict the price (in lakhs).")

# Model selection
#model_choice = st.selectbox("Choose Model", list(models.keys()))
model_choice = st.write("Model used: Random forest")
model = models["Random Forest"]

# User Inputs (raw)
total_sqft = st.number_input("Total Sqft", min_value=500, max_value=10000, step=50)
bhk = st.number_input("BHK (Size)", min_value=1, max_value=10, step=1)
bath = st.number_input("Bathrooms", min_value=1, max_value=10, step=1)
balcony = st.number_input("Balconies", min_value=0, max_value=5, step=1)
location_cleaned = st.selectbox("Location", encoder.categories_[0].tolist())

# -------------------------------
# Preprocess input
# -------------------------------
input_df = pd.DataFrame({
    "total_sqft": [total_sqft],
    "bath": [bath],
    "balcony": [balcony],
    "size_cleared": [bhk],
    "location_cleaned": [location_cleaned]
})

# 1️⃣ Numeric preprocessing: impute + scale
num_features = ["total_sqft", "bath", "balcony", "size_cleared"]
num_data = imputer.transform(input_df[num_features])
num_data = scaler.transform(num_data)

# 2️⃣ Categorical preprocessing: one-hot encode
cat_data = encoder.transform(input_df[["location_cleaned"]])

# 3️⃣ Combine numeric + categorical
X_input = hstack([num_data, cat_data])

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Price"):
    try:
        # Linear Regression requires dense array
        X_input_array = X_input.toarray() if model_choice == "Linear Regression" else X_input
        pred = model.predict(X_input_array)[0]

        # Convert to lakhs (assuming model trained in rupees)
        st.success(f"🏡 Estimated Price: {round(pred, 2)} Lakhs")
    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")

# -------------------------------
# Optional: Show processed vector for debug
# -------------------------------
#if st.checkbox("Show preprocessed input vector"):
#    st.write("Numeric (scaled):", num_data)
#    st.write("Categorical (one-hot):", cat_data.toarray())
#    st.write("Combined input vector:", X_input.toarray())
