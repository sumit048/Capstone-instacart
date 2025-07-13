# main.py

from utils.db_import import load_data_from_db
from pipeline.preprocessing import preprocess_data
from pipeline.training import train_model

# Define DB URL
db_url = "sqlite:///instacart.db"

# Load tables
orders_df = load_data_from_db(db_url, "orders")
products_df = load_data_from_db(db_url, "products")
order_products_df = load_data_from_db(db_url, "order_products__train")

# Preprocess and create features
features_df = preprocess_data(orders_df, products_df, order_products_df)

# Train model if data is valid
if not features_df.empty:
    train_model(features_df, model_path="model.joblib")
else:
    print("❌ Preprocessing failed or returned empty dataset.")
    