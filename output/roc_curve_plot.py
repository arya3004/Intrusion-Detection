import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import joblib
import h5py
import numpy as np

# --- Load model and pipeline ---
model = joblib.load(r".\models\gradient_boost\output\gradient_boost_model.cbm")
pipeline = joblib.load(r".\models\gradient_boost\output\preprocessing_pipeline.pkl")

# --- Load test data ---
with h5py.File(r".\data\test.h5", "r") as f:
    X_test = np.array(f["X"])
    y_test = np.array(f["y"])

# --- Preprocess test data ---
X_test_transformed = pipeline.transform(X_test)

# --- Get predicted probabilities (for ROC curve) ---
y_prob = model.predict_proba(X_test_transformed)[:, 1]

# --- Compute ROC curve and AUC ---
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# --- Plot the ROC curve ---
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - CatBoost Model (ML-IDS)')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
