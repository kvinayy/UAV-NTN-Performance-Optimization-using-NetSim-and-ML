# NOTE:
# This script was used for RF training and analysis
# before integrating the model into the closed-loop pipeline.
# It is not directly invoked by the closed-loop scripts.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset
DATASET = r"C:\Users\Student\Desktop\IMT2023608\UAV\UAV_1_Node\uav_runs_dataset.csv"
df = pd.read_csv(DATASET)

# Clean column names
df.columns = df.columns.str.strip()

print("\nColumns found:", df.columns.tolist(), "\n")

# === Select features (packet size + IAT) ===
X = df[["Packet_Size_Bytes", "IAT_us"]]

# === Target metric: Throughput ===
y = df["Throughput_Mbps"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train Random Forest
model = RandomForestRegressor(
    n_estimators=400,
    random_state=42,
    max_depth=None
)
model.fit(X_train, y_train)

# Validate
pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)

print("=== MODEL TRAINED ===")
print(f"Dataset size: {len(df)} runs")
print(f"Validation MAE: {mae:.4f} Mbps\n")

print("Feature Importances:")
print("Packet_Size_Bytes :", model.feature_importances_[0])
print("IAT_us            :", model.feature_importances_[1])

# Save model
joblib.dump(model, "rf_model.pkl")
print("\nModel saved as rf_model.pkl")
