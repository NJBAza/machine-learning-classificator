import os
import pickle
import sys
from pathlib import Path

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT.parent))
DATAPATH = os.path.join(PACKAGE_ROOT, "data")

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

MODEL_NAME = "classification.pkl"
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT, "trained_models")

TARGET = "priceRange"
MAP = {"250000-350000": 1, "350000-450000": 2, "450000-650000": 3, "650000+": 4, "0-250000": 0}

# loading the conversion dict
with open(os.path.join(DATAPATH, "CONVERSION_DICT"), "rb") as fp:
    CONVERSION_DICT = pickle.load(fp)

# loading the convert features
with open(os.path.join(DATAPATH, "TO_CONVERT"), "rb") as fp:
    TO_CONVERT = pickle.load(fp)

# loading the variables to drop
with open(os.path.join(DATAPATH, "VARIABLES_TO_DROP"), "rb") as fp:
    VARIABLES_TO_DROP = pickle.load(fp)

# loading the categorical features
with open(os.path.join(DATAPATH, "CATEGORICAL_FEATURES"), "rb") as fp:
    CATEGORICAL_FEATURES = pickle.load(fp)

# loading the features to modify
with open(os.path.join(DATAPATH, "FEATURES_MODIFY"), "rb") as fp:
    FEATURES_MODIFY = pickle.load(fp)

# loading the numerical features
with open(os.path.join(DATAPATH, "NUMERICAL_FEATURES"), "rb") as fp:
    NUMERICAL_FEATURES = pickle.load(fp)

# loading the numerical features winsor
with open(os.path.join(DATAPATH, "NUMERICAL_FEATURES_WINSOR"), "rb") as fp:
    NUMERICAL_FEATURES_WINSOR = pickle.load(fp)

# loading the numerical features
with open(os.path.join(DATAPATH, "GEOLOCATION"), "rb") as fp:
    GEOLOCATION = pickle.load(fp)

# loading the description feature
with open(os.path.join(DATAPATH, "DESCRIPTION_FEATURE"), "rb") as fp:
    DESCRIPTION_FEATURE = pickle.load(fp)

# loading the black list
with open(os.path.join(DATAPATH, "BLACK_LIST"), "rb") as fp:
    BLACK_LIST = pickle.load(fp)

# loading the removing words
with open(os.path.join(DATAPATH, "TO_REMOVE"), "rb") as fp:
    TO_REMOVE = pickle.load(fp)
