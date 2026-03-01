import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss
)


def load_data():
    dataset = pd.read_csv("data/synthetic_fraud_dataset.csv")

    # Drop unnecessary columns
    drop_cols = ["Transaction_ID", "Timestamp", "User_ID"]
    dataset = dataset.drop(columns=drop_cols, errors="ignore")

    X = dataset.drop("Fraud_Label", axis=1)
    y = dataset["Fraud_Label"]

    # Convert categorical columns
    categorical_cols = [
        "Location",
        "Transaction_Type",
        "Device_Type",
        "Merchant_Category",
        "Card_Type",
        "Authentication_Method"
    ]

    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype("category")

    return X, y


def train_xgboost(hyperparams: dict):

    X, y = load_data()

    # Train / validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    dtrain = xgb.DMatrix(X_train, y_train, enable_categorical=True)
    dval = xgb.DMatrix(X_val, y_val, enable_categorical=True)

    # Base params always required
    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    }

    params = {**base_params, **hyperparams}

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=hyperparams.get("n_estimators", 100),
        evals=[(dval, "validation")],
        early_stopping_rounds=20,
        verbose_eval=False
    )

    # Predict probabilities
    val_probs = model.predict(dval)
    val_preds = (val_probs > 0.5).astype(int)

    # Compute metrics
    metrics = {
        "val_accuracy": accuracy_score(y_val, val_preds),
        "val_precision": precision_score(y_val, val_preds),
        "val_recall": recall_score(y_val, val_preds),
        "val_f1": f1_score(y_val, val_preds),
        "val_roc_auc": roc_auc_score(y_val, val_probs),
        "val_logloss": log_loss(y_val, val_probs),
        "best_iteration": model.best_iteration
    }

    return metrics

