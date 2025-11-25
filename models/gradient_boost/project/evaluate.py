
import os
import warnings
import click
import joblib
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import mlflow
import matplotlib.pyplot as plt
warnings.simplefilter("default")


def try_load_label_encoder(model_path, pipeline_path):
    """Attempt to load a saved LabelEncoder from model or pipeline directory or repo root."""
    candidates = [
        os.path.join(os.path.dirname(model_path), "label_encoder.pkl") if model_path else None,
        os.path.join(os.path.dirname(pipeline_path), "label_encoder.pkl") if pipeline_path else None,
        os.path.join(os.getcwd(), "label_encoder.pkl")
    ]
    for path in candidates:
        if path and os.path.exists(path):
            try:
                le = joblib.load(path)
                print(f"🔁 Loaded label encoder from: {path}")
                return le, path
            except Exception as e:
                print(f"⚠️ Found encoder file at {path} but failed to load: {e}")
    return None, None


@click.command()
@click.option("--model-path", required=True, help="Path to trained CatBoost model (.cbm).")
@click.option("--pipeline-path", required=True, help="Path to preprocessing pipeline (.pkl).")
@click.option("--test-path", required=True, help="Path to test dataset (.h5).")
def evaluate(model_path, pipeline_path, test_path):
    print("🚀 Loading model, pipeline, and test dataset...")

    # Load model and preprocessing pipeline
    model = CatBoostClassifier()
    model.load_model(model_path)
    preprocessor = joblib.load(pipeline_path)

    # Load test data
    test_df = pd.read_hdf(test_path)
    print(f"✅ Test dataset loaded with {test_df.shape[0]} rows and {test_df.shape[1]} columns.")

    # Standardize label column
    if "Label" not in test_df.columns and "label" in test_df.columns:
        test_df.rename(columns={"label": "Label"}, inplace=True)
    if "Label" not in test_df.columns:
        raise ValueError("No 'Label' column in test dataset. Found columns: " + ", ".join(test_df.columns))

    print("\n🔎 Test label distribution:")
    print(test_df["Label"].value_counts())

    # Separate features and labels
    X_test = test_df.drop(columns=["Label"])
    y_test = test_df["Label"]

    # Handle invalid values
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.fillna(0, inplace=True)

    # Check pipeline feature names
    preproc_cols = getattr(preprocessor, "feature_names_in_", None)
    if preproc_cols is not None:
        if list(preproc_cols) != list(X_test.columns):
            print("\n⚠️ Feature name/order mismatch between pipeline and test data.")
            print(" - pipeline (first 10):", list(preproc_cols)[:10])
            print(" - test (first 10):", list(X_test.columns)[:10])
        else:
            print("\n✅ Preprocessor feature names match test data.")
    else:
        print("\nℹ️ Preprocessor does not expose feature_names_in_. Ensure columns match training.")
        print("Test columns count:", X_test.shape[1])

    # Transform test features
    print("\n🔄 Applying preprocessing pipeline...")
    X_test_preprocessed = preprocessor.transform(X_test)

    # Predict
    print("\n⚙️ Running predictions...")
    y_pred = np.asarray(model.predict(X_test_preprocessed))
    uniq_pred, counts_pred = np.unique(y_pred, return_counts=True)
    print("\n🔎 Prediction distribution:")
    for u, c in zip(uniq_pred, counts_pred):
        print(f"  {u!r}: {c}")

    # Attempt to load label encoder
    le_loaded, encoder_path = try_load_label_encoder(model_path, pipeline_path)

    # Prepare holders for downstream metrics
    y_pred_for_metrics = None
    y_test_for_metrics = None
    classes_mapping_info = {}

    # Determine prediction type
    if y_pred.dtype.kind in {"U", "S", "O"}:
        # Predictions are string labels
        y_pred_for_metrics = y_pred.astype(str)
        y_test_for_metrics = y_test.astype(str).values
        classes_mapping_info = {"mode": "string_preds"}
    else:
        # Predictions are numeric
        if le_loaded:
            try:
                y_pred_for_metrics = le_loaded.inverse_transform(y_pred.astype(int))
                y_test_for_metrics = y_test.astype(str).values
                classes_mapping_info = {"mode": "encoder_loaded", "encoder_path": encoder_path}
                print(f"\n🔁 Converted numeric predictions to strings using encoder at {encoder_path}")
            except Exception:
                y_pred_for_metrics = None
        if y_pred_for_metrics is None:
            # Fit LabelEncoder on test labels
            le_test = LabelEncoder()
            y_test_encoded = le_test.fit_transform(y_test.astype(str))
            mapping = dict(zip(le_test.classes_, range(len(le_test.classes_))))
            print("\n⚠️ No saved encoder. Fitted LabelEncoder on test labels:", mapping)
            if np.max(y_pred) < len(le_test.classes_):
                y_pred_for_metrics = y_pred.astype(int)
                y_test_for_metrics = y_test_encoded
                classes_mapping_info = {"mode": "fitted_on_test", "mapping": mapping}
            else:
                # Map predictions to closest class
                y_pred_for_metrics = np.array([le_test.classes_[v] if 0 <= v < len(le_test.classes_) else str(v)
                                               for v in y_pred.astype(int)])
                y_test_for_metrics = y_test.astype(str).values
                classes_mapping_info = {"mode": "mapped_partial", "mapping": mapping}

    # Compute binary metrics: Benign=0, Attack=1
    # y_true binary based on original string labels
    y_test_str = y_test.astype(str).values
    y_test_binary = np.where(y_test_str == "Benign", 0, 1)

    # y_pred binary depending on representation used above
    if isinstance(y_pred_for_metrics[0], (str, np.str_)):
        y_pred_binary = np.where(np.array(y_pred_for_metrics, dtype=str) == "Benign", 0, 1)
    else:
        # numeric predictions; find the index corresponding to "Benign"
        benign_idx = None
        if isinstance(classes_mapping_info, dict) and "mapping" in classes_mapping_info:
            benign_idx = classes_mapping_info["mapping"].get("Benign")
        if benign_idx is None and le_loaded is not None:
            try:
                benign_idx = int(np.where(le_loaded.classes_ == "Benign")[0][0])
            except Exception:
                benign_idx = 0
        if benign_idx is None:
            benign_idx = 0
        y_pred_binary = np.where(np.asarray(y_pred_for_metrics, dtype=int) == benign_idx, 0, 1)

    # ROC-AUC (Attack=1 probability)
    proba = model.predict_proba(X_test_preprocessed)
    if proba.ndim == 1:
        y_score = proba
    else:
        # Identify the probability column for Benign
        benign_idx_roc = None
        if isinstance(classes_mapping_info, dict) and "mapping" in classes_mapping_info:
            benign_idx_roc = classes_mapping_info["mapping"].get("Benign")
        if benign_idx_roc is None and le_loaded is not None:
            try:
                benign_idx_roc = int(np.where(le_loaded.classes_ == "Benign")[0][0])
            except Exception:
                benign_idx_roc = 0
        if benign_idx_roc is None:
            benign_idx_roc = 0
        # Attack probability = 1 - P(Benign)
        y_score = 1.0 - proba[:, benign_idx_roc]

    fpr, tpr, thresholds = roc_curve(y_test_binary, y_score)
    auc_bin = roc_auc_score(y_test_binary, y_score)

    # Save ROC curve plot next to model
    out_dir = os.path.dirname(model_path) if model_path else os.getcwd()
    roc_path = os.path.join(out_dir, "roc_curve_binary.png")
    try:
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"ROC (AUC = {auc_bin:.4f})")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Benign vs Attack)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(roc_path, dpi=150)
        plt.close()
        print(f"Saved ROC curve to: {roc_path}")
    except Exception as e:
        print(f"⚠️ Failed to save ROC curve plot: {e}")

    cm_2x2 = confusion_matrix(y_test_binary, y_pred_binary, labels=[0, 1])
    acc_bin = accuracy_score(y_test_binary, y_pred_binary)
    report_bin = classification_report(
        y_test_binary,
        y_pred_binary,
        labels=[0, 1],
        target_names=["Benign(0)", "Attack(1)"],
        zero_division=0,
    )

    # Print results (binary-only)
    print("\n📊 Model Evaluation Results (Binary)")
    print("🧾 Classes mapping info:", classes_mapping_info)
    print(f"\nAccuracy: {acc_bin:.4f}")
    print(f"ROC-AUC: {auc_bin:.4f}\n")
    print("Classification Report (0=Benign, 1=Attack):\n", report_bin)
    print("2x2 Confusion Matrix (Benign=0, Attack=1):\n", cm_2x2)
    print("\nMatrix layout: [[TN, FP], [FN, TP]]")

    # Log metrics to MLflow (binary only)
    with mlflow.start_run(run_name="model_evaluation"):
        mlflow.log_metric("test_accuracy_binary", float(acc_bin))
        mlflow.log_metric("roc_auc_binary", float(auc_bin))
        mlflow.log_text(report_bin, "classification_report_binary.txt")
        mlflow.log_text(str(cm_2x2.tolist()), "confusion_matrix_2x2.json")
        try:
            mlflow.log_artifact(roc_path, artifact_path="plots")
        except Exception as e:
            print(f"⚠️ Failed to log ROC curve to MLflow: {e}")

    print("\n✅ Evaluation complete. Binary metrics and ROC logged to MLflow.")


if __name__ == "__main__":
    evaluate()
