# prepare_data.py

import pandas as pd
import os

# ✅ Load only a limited number of rows to reduce size (adjust as needed)
orders = pd.read_csv("data/orders.csv").query("user_id < 1000")
order_products_prior = pd.read_csv("data/order_products__prior.csv").head(10000)
products = pd.read_csv("data/products.csv")

print("✅ Sample datasets loaded successfully.")

# ✅ Merge orders and order_products_prior
merged_df = pd.merge(order_products_prior, orders, on="order_id", how="inner")

# ✅ Add product names
merged_df = pd.merge(merged_df, products[['product_id', 'product_name']], on='product_id', how='left')

# ✅ Feature: Product reorder ratio
product_reorder_ratio = merged_df.groupby("product_id")["reordered"].mean().reset_index()
product_reorder_ratio.columns = ["product_id", "product_reorder_ratio"]

merged_df = pd.merge(merged_df, product_reorder_ratio, on="product_id", how="left")

# ✅ Feature: Total products in each order
order_size = merged_df.groupby("order_id")["product_id"].count().reset_index()
order_size.columns = ["order_id", "order_product_count"]

merged_df = pd.merge(merged_df, order_size, on="order_id", how="left")

# ✅ Keep only selected columns (used in training)
final_df = merged_df[[ 
    "add_to_cart_order", 
    "reordered", 
    "order_product_count", 
    "product_name"
]]

# ✅ Save to artifacts folder
os.makedirs("artifacts", exist_ok=True)
final_df.to_csv("artifacts/cleaned_data.csv", index=False)

print("✅ Cleaned data saved to: artifacts/cleaned_data.csv")
print(f"✅ Final dataset shape: {final_df.shape}")
