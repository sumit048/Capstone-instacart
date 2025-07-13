import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
from utils.db_import import load_data_from_db
from sklearn.preprocessing import LabelEncoder

# ------------------ PAGE CONFIGURATION ------------------
st.set_page_config(page_title="Instacart Reorder Predictor", layout="wide")

# ------------------ CUSTOM GLOBAL STYLES ------------------
st.markdown("""
<style>
...  # (KEEP ALL YOUR CUSTOM CSS UNCHANGED HERE)
</style>
""", unsafe_allow_html=True)

# ------------------ LOGO SECTION ------------------
def image_to_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

encoded_logo = image_to_base64("instacart_logo.png")

st.markdown(f"""
<div id="logo-container">
    <img src="data:image/png;base64,{encoded_logo}" width="180"/>
</div>
""", unsafe_allow_html=True)

# ------------------ LOAD TRAINED MODEL ------------------
model = joblib.load("model.joblib")

# ------------------ SETUP PAGE STATE ------------------
if "page" not in st.session_state:
    st.session_state.page = "single"

# ------------------ PAGE TITLE ------------------
st.markdown("<div style='padding-top: 80px;'></div>", unsafe_allow_html=True)
st.markdown("<h1 style='color: #FFF4CB; font-weight: 800;'>🍒 Instacart Reorder Prediction Dashboard</h1>", unsafe_allow_html=True)

# ------------------ PAGE SWITCHING BUTTONS ------------------
col1, col2 = st.columns([1, 1])
with col1:
    if st.session_state.page == "single":
        st.markdown("<div class='toggle-button active'>🍇 Single Prediction</div>", unsafe_allow_html=True)
    else:
        if st.button("🍇 Single Prediction", key="to_single"):
            st.session_state.page = "single"
            st.rerun()

with col2:
    if st.session_state.page == "batch":
        st.markdown("<div class='toggle-button active'>📂 Batch Prediction</div>", unsafe_allow_html=True)
    else:
        if st.button("📂 Batch Prediction", key="to_batch"):
            st.session_state.page = "batch"
            st.rerun()

# ------------------ SINGLE PREDICTION MODE ------------------
if st.session_state.page == "single":
    st.subheader("Enter Order Details")

    user_id = st.number_input("User ID", min_value=1, value=1)
    products_df = load_data_from_db("sqlite:///instacart.db", "products")
  

        # Debug: show what columns were actually loaded
    st.write("📋 Columns in products_df:", products_df.columns.tolist())

    # Safely pick column name for product_name
    if "product_name" in products_df.columns:
        product_name_col = "product_name"
    elif "name" in products_df.columns:
        product_name_col = "name"
    else:
        st.error("❌ Could not find product name column in products table.")
        st.stop()

    # Check product_id exists
    if "product_id" not in products_df.columns:
        st.error("❌ 'product_id' column not found.")
        st.stop()

    # Create product map
    product_map = dict(zip(products_df[product_name_col], products_df["product_id"]))



    product_name = st.selectbox("Product Name", list(product_map.keys()))
    product_id = product_map[product_name]
    st.write(f"Selected Product ID: `{product_id}`")

    order_dow = st.selectbox("Day of Week", list(range(7)), format_func=lambda x: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][x])
    order_hour_of_day = st.slider("Order Hour", 0, 23, 10)
    add_to_cart_order = st.number_input("Add-to-Cart Position", min_value=1, value=1)
    user_total_orders = st.number_input("User Total Orders", min_value=1, value=5)
    product_reorder_rate = st.slider("Product Reorder Rate", min_value=0.0, max_value=1.0, value=0.3)
    days_since_prior_order = st.slider("Days Since Prior Order", min_value=0, max_value=30, value=7)

    le_product = LabelEncoder()
    le_product.fit(products_df["product_name"].astype(str))
    product_name_encoded = le_product.transform([product_name])[0]

    if user_id > 200000 or product_id > 50000:
        st.error("❌ Unknown user or product – reorder cannot be predicted reliably.")
    else:
        if st.button("🔍 Predict Reorder"):
            input_data = pd.DataFrame({
                "user_id": [user_id],
                "product_id": [product_id],
                "product_name_encoded": [product_name_encoded],
                "order_dow": [order_dow],
                "order_hour_of_day": [order_hour_of_day],
                "add_to_cart_order": [add_to_cart_order],
                "user_total_orders": [user_total_orders],
                "product_reorder_rate": [product_reorder_rate],
                "days_since_prior_order": [days_since_prior_order]
            })
            prediction = model.predict(input_data)[0]

            if prediction == 1:
                st.markdown("""<div style="background-color:#22c55e; padding:20px; border-radius:12px; text-align:center; animation: fadeIn 0.7s ease-out;">
                    <img src="https://cdn-icons-png.flaticon.com/512/845/845646.png" width="80"/>
                    <h3 style="color:white; margin-top:10px;">Product likely to be reordered</h3>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div style="background-color:#dc2626; padding:20px; border-radius:12px; text-align:center; animation: fadeIn 0.7s ease-out;">
                    <img src="https://cdn-icons-png.flaticon.com/512/463/463612.png" width="80"/>
                    <h3 style="color:white; margin-top:10px;">Product not likely to be reordered</h3>
                </div>""", unsafe_allow_html=True)

# ------------------ BATCH PREDICTION MODE ------------------
elif st.session_state.page == "batch":
    st.subheader("Upload a CSV of Orders 📂")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        base_cols = [
            "user_id", "product_name", "order_dow", "order_hour_of_day",
            "add_to_cart_order", "user_total_orders", "product_reorder_rate",
            "days_since_prior_order"
        ]

        missing_cols = [col for col in base_cols if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Missing columns in uploaded file: {missing_cols}")
            st.stop()

        products_df = load_data_from_db("sqlite:///instacart.db", "products")
        le_product = LabelEncoder()
        le_product.fit(products_df["product_name"].astype(str))

        product_map = dict(zip(products_df["product_name"], products_df["product_id"]))
        df["product_id"] = df["product_name"].map(product_map)
        df["product_name_encoded"] = le_product.transform(df["product_name"].astype(str))

        if "order_product_count" not in df.columns:
            df["order_product_count"] = 5

        final_cols = [
            "user_id", "product_id", "product_name_encoded",
            "order_dow", "order_hour_of_day", "add_to_cart_order",
            "user_total_orders", "product_reorder_rate",
            "days_since_prior_order"
        ]

        try:
            input_data = df[final_cols]
            predictions = model.predict(input_data)
            df["reordered_prediction"] = predictions
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.stop()

        yes = int((df["reordered_prediction"] == 1).sum())
        no = int((df["reordered_prediction"] == 0).sum())

        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Will Reorder", yes)
        col2.metric("❌ Will Not Reorder", no)
        col3.metric("📦 Total Rows", len(df))

        st.dataframe(df)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results", csv, "instacart_predictions.csv", "text/csv")
