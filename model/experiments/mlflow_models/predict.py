import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(str(PACKAGE_ROOT.parent))

from config import config


def predictions(data_input, pipeline=None):
    data = pd.DataFrame(data_input)
    FEATURES = list(data.columns)
    FEATURES.remove(config.TARGET)
    pred = pipeline.predict(data[FEATURES])
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


# def predictions():
#     test_data = load_dataset(config.TEST_FILE)
#     FEATURES = list(test_data.columns)
#     FEATURES.remove(config.TARGET)
#     pred = classification_pipeline.predict(test_data[FEATURES])
#     output = np.where(
#         pred == 1,
#         "250000-350000",
#         np.where(
#             pred == 2,
#             "350000-450000",
#             np.where(
#                 pred == 3,
#                 "450000-650000",
#                 np.where(pred == 4, "650000+", "0-250000"),
#             ),
#         ),
#     )
#     # result = {"Predictions": output}
#     return print(output)


if __name__ == "__main__":
    predictions()
