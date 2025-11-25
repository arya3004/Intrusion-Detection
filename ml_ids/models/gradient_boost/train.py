"""
Utilities to train a machine learning estimator based on the Gradient Boosting algorithm using the CatBoost library.
"""

import logging
from collections import namedtuple
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import FunctionTransformer

from ml_ids.transform.preprocessing import create_pipeline
from ml_ids.transform.sampling import upsample_minority_classes
from ml_ids.model_selection import split_x_y

LOGGER = logging.getLogger(__name__)

GradientBoostHyperParams = namedtuple('GradientBoostHyperParams',
                                      ['nr_iterations', 'tree_depth', 'l2_reg', 'border_count', 'random_strength',
                                       'task_type'])


def fit_pipeline(train_dataset):
    """
    Creates and fits the scikit-learn pre-processing pipeline.
    """
    cols_to_impute = train_dataset.columns[train_dataset.isna().any()].tolist()

    X_train, _ = split_x_y(train_dataset)
    pipeline, get_col_names = create_pipeline(X_train)
    pipeline.fit(X_train)

    print("✅ Pipeline fitted successfully.")
    return pipeline, get_col_names


def preprocess_val_dataset(pipeline, val_dataset):
    """
    Pre-processes the validation dataset.
    """
    X_val, y_val = split_x_y(val_dataset)
    X_val = pipeline.transform(X_val)
    return X_val, y_val.label_is_attack


def preprocess_train_dataset(pipeline, train_dataset, nr_attack_samples, random_state):
    """
    Pre-processes the training dataset.
    """
    X_train, y_train = split_x_y(train_dataset)
    X_train = pipeline.transform(X_train)

    X_train, y_train = upsample_minority_classes(
        X_train, y_train, min_samples=nr_attack_samples, random_state=random_state
    )

    return X_train, (y_train != 0).astype('int')


def calculate_class_weights(y_train):
    """
    Calculates the class weights of the unique classes in the training labels.
    """
    minority_class_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    return [1, minority_class_weight]


def train_gb_classifier(train_pool, val_pool, class_weights,
                        nr_iterations, tree_depth, l2_reg, border_count,
                        random_strength, task_type, random_state=None):
    """
    Trains an estimator based on the Gradient Boosting algorithm using the CatBoost library.
    """
    clf = CatBoostClassifier(
        loss_function='Logloss',
        iterations=nr_iterations,
        depth=tree_depth,
        l2_leaf_reg=l2_reg,
        border_count=border_count,
        random_strength=random_strength,
        task_type=task_type,
        class_weights=class_weights,
        verbose=1,
        random_seed=random_state
    )

    clf.fit(train_pool, eval_set=val_pool)
    return clf


def train_model(train_dataset: pd.DataFrame,
                val_dataset: pd.DataFrame,
                hyper_params: GradientBoostHyperParams,
                nr_attack_samples: int,
                random_seed: int = None):
    """
    Trains an estimator based on the Gradient Boosting algorithm using the CatBoost library.
    """
    LOGGER.info('Training model with parameters [samples-per-attack-category=%s, hyperparams=%s]',
                nr_attack_samples, hyper_params)

    pipeline, col_names = fit_pipeline(train_dataset)
    X_train, y_train = preprocess_train_dataset(pipeline, train_dataset, nr_attack_samples, random_seed)
    train_pool = Pool(X_train, y_train)

    if val_dataset is not None:
        X_val, y_val = preprocess_val_dataset(pipeline, val_dataset)
        val_pool = Pool(X_val, y_val)
    else:
        val_pool = None

    clf = train_gb_classifier(
        train_pool=train_pool,
        val_pool=val_pool,
        class_weights=calculate_class_weights(y_train),
        nr_iterations=hyper_params.nr_iterations,
        tree_depth=hyper_params.tree_depth,
        l2_reg=hyper_params.l2_reg,
        border_count=hyper_params.border_count,
        random_strength=hyper_params.random_strength,
        task_type=hyper_params.task_type,
        random_state=random_seed
    )

    return clf, pipeline, col_names


# ============================================================
# ✅ MAIN EXECUTION STARTS HERE
# ============================================================
if __name__ == "__main__":
    import os
    import joblib

    print("🚀 Starting training process...")

    # ✅ Load training and validation datasets
    train_dataset = pd.read_hdf("data/train.h5")
    val_dataset = pd.read_hdf("data/val.h5")

    # ✅ Define hyperparameters
    hyper_params = GradientBoostHyperParams(
        nr_iterations=500,
        tree_depth=8,
        l2_reg=3,
        border_count=128,
        random_strength=1,
        task_type="CPU"  # Change to "GPU" if CUDA is available
    )

    # ✅ Train model
    model, pipeline, col_names = train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        hyper_params=hyper_params,
        nr_attack_samples=5000,
        random_seed=42
    )

    # ✅ Save trained model and preprocessing pipeline
    os.makedirs("ml_ids/models/gradient_boost/output", exist_ok=True)
    model.save_model("ml_ids/models/gradient_boost/output/gradient_boost_model.cbm")
    joblib.dump(pipeline, "ml_ids/models/gradient_boost/output/preprocessing_pipeline.pkl")

    print("✅ Model and pipeline saved successfully!")
