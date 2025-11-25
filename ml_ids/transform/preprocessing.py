import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


def load_data(path):
    """Load data from HDF5 file"""
    df = pd.read_hdf(path)
    return df


def save_data(df, path):
    """Save processed DataFrame to HDF5"""
    df.to_hdf(path, key="df", mode="w")


def create_pipeline(X_train=None, imputer_strategy="mean"):
    """
    Creates a preprocessing pipeline that handles both numeric and categorical data.
    Allows customization of the imputer strategy (default: mean).
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer

    # Identify numeric and categorical columns automatically
    if X_train is not None:PIP
        num_cols = X_train.select_dtypes(include=["number"]).columns
        cat_cols = X_train.select_dtypes(exclude=["number"]).columns
    else:
        num_cols, cat_cols = [], []

    # Pipelines for numeric and categorical data
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=imputer_strategy)),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine into a single preprocessor
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    return preprocessor, list(num_cols) + list(cat_cols)

def fit_preprocessor(df):
    """
    Fits the preprocessing pipeline to the dataset.
    """
    import numpy as np
    import os
    import joblib
    from sklearn.pipeline import Pipeline

    # Replace infinities and very large numbers with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Clip only numeric columns safely
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].clip(lower=-1e9, upper=1e9)

    # Create and fit preprocessing pipeline
    preprocessor, feature_names = create_pipeline(df)
    preprocessor.fit(df)

    # ✅ Ensure artifacts folder exists before saving
    os.makedirs("ml_ids/artifacts", exist_ok=True)

    # Save preprocessor for reuse
    joblib.dump(preprocessor, "ml_ids/artifacts/preprocessor.pkl")
    print("✅ Preprocessor fitted and saved successfully.")

    return preprocessor



def transform_data(df, preprocessor_path="models/transform/preprocessor.joblib"):
    """Load the preprocessor and transform new data"""
    preprocessor = joblib.load(preprocessor_path)
    X = preprocessor.transform(df)
    print(f"✅ Data transformed - shape: {X.shape}")
    return X


def get_feature_names(preprocessor, num_cols=None, cat_cols=None):
    """Get combined feature names for numeric + one-hot encoded categorical features"""
    # This function is optional for your simpler pipeline (no OneHotEncoder)
    if hasattr(preprocessor, "transformers_"):
        num_names = num_cols or []
        cat_names = preprocessor.transformers_[1][1].named_steps["encoder"].get_feature_names_out(cat_cols)
        all_features = np.concatenate([num_names, cat_names])
        return all_features
    else:
        return ["feature_" + str(i) for i in range(preprocessor.transform(np.zeros((1, len(df.columns)))).shape[1])]


if __name__ == "__main__":
    path = "data/train.h5"
    df = load_data(path)

    preprocessor = fit_preprocessor(df)
    print("✅ Preprocessor training complete.")
