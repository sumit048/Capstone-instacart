import pandas as pd
import sqlite3

# ✅ Load your cleaned CSV
df = pd.read_csv("artifacts/cleaned_data.csv")

# ✅ Connect to SQLite DB (will create if not exists)
conn = sqlite3.connect("instacart.db")

# ✅ Save to table named 'products' (required by your app)
df.to_sql("products", conn, index=False, if_exists="replace")

conn.close()

print("✅ instacart.db created with table: 'products'")
