import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(str(PACKAGE_ROOT.parent))

import prediction_model.processing.preprocessing as pp
from prediction_model.config import config
from prediction_model.pipeline import classification_pipeline
from prediction_model.processing.data_handling import (
    load_dataset,
    save_pipeline,
)


def perform_training():
    train_data = load_dataset(config.TRAIN_FILE)
    train_y = train_data[config.TARGET].map(config.MAP)
    FEATURES = list(train_data.columns)
    FEATURES.remove(config.TARGET)
    classification_pipeline.fit(train_data[FEATURES], train_y)
    # classification_pipeline.fit(train_data)
    save_pipeline(classification_pipeline)


if __name__ == "__main__":
    perform_training()
