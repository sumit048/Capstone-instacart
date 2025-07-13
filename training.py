# pipeline/training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
import os

def train_model(df, model_path="artifacts/model.joblib"):
    """
    Train a Random Forest classifier to predict 'reordered'.
    """
    try:
        logging.info("Preparing training data...")

        # Drop missing values
        df = df.dropna()

        # ✅ Encode categorical column (product_name)
        if "product_name" in df.columns:
            le = LabelEncoder()
            df["product_name_encoded"] = le.fit_transform(df["product_name"])
            df.drop(columns=["product_name"], inplace=True)

        # Ensure 'reordered' exists
        if "reordered" not in df.columns:
            logging.error("'reordered' column not found in data.")
            return

        # Separate features and target
        X = df.drop("reordered", axis=1)
        y = df["reordered"]

        # ✅ Log features
        print("🧠 Features used for training:", X.columns.tolist())
        print("✅ Final data types:\n", X.dtypes)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        logging.info("Training RandomForest model...")

        # ✅ Faster training config
        clf = RandomForestClassifier(
            n_estimators=10,            # ⬅️ Reduced from 100
            max_depth=10,
            max_features="sqrt",
            random_state=42
        )

        clf.fit(X_train, y_train)

        # Evaluate
        y_pred = clf.predict(X_test)
        print("📊 Classification Report:\n", classification_report(y_test, y_pred))
        print("✅ Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        os.makedirs("artifacts", exist_ok=True)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Not Reordered", "Reordered"],
                    yticklabels=["Not Reordered", "Reordered"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("artifacts/confusion_matrix.png")
        plt.show()

        # Save model
        joblib.dump(clf, model_path)
        logging.info(f"✅ Model saved to: {model_path}")

    except Exception as e:
        logging.error(f"Training failed: {e}")
        print(f"❌ Training failed: {e}")

# ✅ Main execution
if __name__ == "__main__":
    # Load only 10,000 rows for faster training
    df = pd.read_csv("../artifacts/cleaned_data.csv", nrows=10000)
    train_model(df, model_path="artifacts/model.joblib")
