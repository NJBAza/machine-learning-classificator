import os
import sys
from pathlib import Path

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Assuming 'prediction_model' is in the parent directory
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT.parent))

import processing.preprocessing as pp
from config import config

pipeline_features = Pipeline(
    [
        (
            "DataFrameTypeConverter",
            pp.DataFrameTypeConverter(conversion_dict=config.CONVERSION_DICT),
        ),
        (
            "DropColumns",
            pp.DropColumns(variables_to_drop=config.VARIABLES_TO_DROP),
        ),
        (
            "ModeImputer",
            pp.ModeImputer(variables=config.CATEGORICAL_FEATURES),
        ),
        (
            "DataFrameProcessor",
            pp.DataFrameProcessor(features=config.FEATURES_MODIFY, quantile_threshold=0.1),
        ),
        (
            "MedianImputer",
            pp.MedianImputer(variables=config.NUMERICAL_FEATURES),
        ),
        (
            "Winsorizer",
            pp.Winsorizer(numerical_features=config.NUMERICAL_FEATURES, limits=[0.025, 0.025]),
        ),
        (
            "DropColumns2",
            pp.DropColumns(variables_to_drop=config.NUMERICAL_FEATURES),
        ),
        (
            "CoordinateBinner",
            pp.CoordinateBinner(columns=config.GEOLOCATION, decimal_places=3),
        ),
        (
            "DropColumns3",
            pp.DropColumns(variables_to_drop=config.GEOLOCATION),
        ),
        (
            "TextProcessor",
            pp.TextProcessor(column=config.DESCRIPTION_FEATURE, blacklist=config.BLACK_LIST),
        ),
        (
            "TFIDFTransformer",
            pp.TFIDFTransformer(column=config.DESCRIPTION_FEATURE, max_features=20),
        ),
        (
            "NumericalFeatureSelector",
            pp.NumericalFeatureSelector(remove_columns=config.TO_REMOVE),
        ),
        (
            "LogTransforms",
            pp.LogTransforms(),
        ),
        (
            "DataScaler",
            pp.DataScaler(),
        ),
        (
            "CorrelationMatrixProcessor",
            pp.CorrelationMatrixProcessor(threshold=0.7),
        ),
        (
            "FeatureVariance",
            pp.FeatureVariance(threshold=0.001),
        ),
        (
            "DropColumns4",
            pp.DropColumns(variables_to_drop=config.DESCRIPTION_FEATURE),
        ),
        (
            "LabelEncoderProcessor",
            pp.LabelEncoderProcessor(),
        ),
    ]
)
