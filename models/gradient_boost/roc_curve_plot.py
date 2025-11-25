import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import joblib
import h5py
import numpy as np
import pandas as pd

# --- Load model and pipeline ---
from catboost import CatBoostClassifier
model = CatBoostClassifier()
model.load_model(r".\models\gradient_boost\output\gradient_boost_model.cbm")
pipeline = joblib.load(r".\models\gradient_boost\output\preprocessing_pipeline.pkl")

# --- Load test data ---
with h5py.File(r".\data\test.h5", "r") as f:
    df = np.array(f["df"])

print("Type of df:", type(df))
print("Shape of df:", getattr(df, "shape", "no shape"))
print("Dtype of df:", getattr(df, "dtype", "no dtype"))

# --- Flatten structured array (if needed) ---
if len(df.shape) == 1 and hasattr(df[0], 'tolist'):
    df = np.array([list(x) for x in df])

print("After flattening → Shape:", df.shape)

# --- Split features and labels ---
X_test = df[:, :-1]
y_test = df[:, -1]

# --- Preprocess ---
X_test_transformed = pipeline.transform(X_test)

# --- Predict probabilities ---
y_prob = model.predict_proba(X_test_transformed)[:, 1]

# --- Compute ROC and AUC ---
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# --- Plot ---
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - CatBoost Model (ML-IDS)')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
