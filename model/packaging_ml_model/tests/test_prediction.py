# Machine learning code to test and best practices for writing those tests.
# ***What to Test:***<br/>
# **Inputs:**
# 1. Data types (numbers, strings, etc.)
# 2. Format (valid structure)
# 3. Length (within expected range)
# 4. Edge cases (minimum, maximum, very small, very large values)

# **Outputs:**
# 1. Data types (numbers, strings, etc.)
# 2. Format (valid structure)
# 3. Exceptions (handling errors gracefully)
# 4. Both intermediate and final outputs

import os
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(str(PACKAGE_ROOT.parent))
print(PACKAGE_ROOT)


from prediction_model.config import config
from prediction_model.predict import predictions
from prediction_model.processing.data_handling import load_dataset


@pytest.fixture
def single_prediction():
    test_data = load_dataset(config.TEST_FILE)
    single_row = test_data[:1]
    return predictions(single_row)


def test_single_prediction_not_none(single_prediction):
    assert single_prediction is not None


def test_single_prediction_str_type(single_prediction):
    assert isinstance(single_prediction["Predictions"][0], str)


def test_single_prediction_validation(single_prediction):
    assert single_prediction["Predictions"][0] == "350000-450000"


# import numpy as np

# algo = {"Predictions": np.array(["350000-450000"], dtype="<U13")}

# print(algo["Predictions"][0])
