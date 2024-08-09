import datetime as dt
import glob
import os
import pickle
import re
import string
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import featuretools as ft
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk import word_tokenize
from nltk.corpus import (
    stopwords,
    wordnet,
)
from nltk.stem import (
    PorterStemmer,
    WordNetLemmatizer,
)
from scipy.stats.mstats import winsorize
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    StandardScaler,
    normalize,
)
from tqdm import tqdm

nltk.download("wordnet")
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)
