import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT.parent))

import processing.preprocessing as pp
from config import config
from pipeline_features import pipeline_features
from predict import predictions
from processing.data_handling import load_dataset

RANDOM_SEED = 20230916


def perform_training(model=None):
    train_data = load_dataset(config.TRAIN_FILE)
    train_y = train_data[config.TARGET].map(config.MAP)
    FEATURES = list(train_data.columns)
    FEATURES.remove(config.TARGET)
    full_pipeline = Pipeline(
        steps=[("pipeline_features", pipeline_features), ("classifier", model)]
    )
    full_pipeline.fit(train_data[FEATURES], train_y)
    # classification_pipeline.fit(train_data)
    return full_pipeline


# RandomForest
rf = RandomForestClassifier(random_state=RANDOM_SEED)
param_grid_forest = {
    "n_estimators": [100],  # [100, 200, 300],
    "max_depth": [5],  # [5, 10, 15],
    # "criterion": ["gini", "entropy"],
    # "max_leaf_nodes": 50,  # [50, 100],
}

grid_forest = GridSearchCV(
    estimator=rf, param_grid=param_grid_forest, cv=5, n_jobs=-1, scoring="accuracy", verbose=0
)
model_forest = perform_training(grid_forest)

# XGBoost Classifier

xgb_classifier = XGBClassifier(random_state=RANDOM_SEED)
param_grid_xgboost = {
    "n_estimators": [100],  # [100, 200, 300],
    "learning_rate": [0.01],  # [0.01, 0.1, 0.5],
    # "max_depth": 5,  # [5, 10],
    # "reg_alpha": 0.01,  # [0.01, 0.1],
    # "reg_lambda": 0.01,  # [0.01, 0.1],
}

grid_xgboost = GridSearchCV(
    estimator=xgb_classifier,
    param_grid=param_grid_xgboost,
    cv=5,
    n_jobs=-1,
    scoring="accuracy",
    verbose=0,
)
model_xgboost = perform_training(grid_xgboost)
mlflow.set_experiment("Houses Price Range")


def predictions(data_input, model=None):
    data = pd.DataFrame(data_input)
    FEATURES = list(data.columns)
    FEATURES.remove(config.TARGET)
    full_pipeline = Pipeline(
        steps=[("pipeline_features", pipeline_features), ("classifier", model)]
    )
    pred = full_pipeline.predict(data[FEATURES])
    output = np.where(
        pred == 1,
        "250000-350000",
        np.where(
            pred == 2,
            "350000-450000",
            np.where(
                pred == 3,
                "450000-650000",
                np.where(pred == 4, "650000+", "0-250000"),
            ),
        ),
    )
    result = {"Predictions": output}
    return result


def single_prediction():
    test_data = load_dataset(config.TEST_FILE)
    single_row = test_data[:1]
    return predictions(single_row)


def mlflow_logging(model, data_input, name):

    with mlflow.start_run() as run:
        #        mlflow.set_tracking_uri("http://0.0.0.0:5001/")
        run_id = run.info.run_id
        mlflow.set_tag("run_id", run_id)
        data = pd.DataFrame(data_input)
        FEATURES = list(data.columns)
        FEATURES.remove(config.TARGET)
        full_pipeline = Pipeline(
            steps=[("pipeline_features", pipeline_features), ("classifier", model)]
        )
        pred = full_pipeline.predict(data[FEATURES])
        # metrics
        accuracy = accuracy_score(data[config.TARGET], pred)
        f1 = f1_score(data[config.TARGET], pred)
        auc = roc_auc_score(data[config.TARGET], pred)
        # Logging best parameters from gridsearch
        mlflow.log_params(model.best_params_)
        # log the metrics
        mlflow.log_metric("Mean CV score", model.best_score_)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("f1-score", f1)
        mlflow.log_metric("AUC", auc)

        # Logging artifacts and model
        mlflow.log_artifact("plots/ROC_curve.png")
        mlflow.sklearn.log_model(model, name)

        mlflow.end_run()


mlflow_logging(model_forest, load_dataset(config.TEST_FILE), "RandomForestClassifier")
mlflow_logging(model_xgboost, load_dataset(config.TEST_FILE), "XGBClassifier")
