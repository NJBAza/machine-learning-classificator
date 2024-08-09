import os
import re
import sys
from pathlib import Path

import featuretools as ft
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from scipy.stats.mstats import winsorize
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    StandardScaler,
    normalize,
)

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT.parent))


# datatype converter
class DataFrameTypeConverter(BaseEstimator, TransformerMixin):
    def __init__(self, conversion_dict):
        self.conversion_dict = conversion_dict

    def fit(self, X, y=None):
        return self  # Nothing to fit, so just return self

    def transform(self, X):
        X = X.copy()
        for column, new_type in self.conversion_dict.items():
            X[column] = X[column].astype(new_type)
        return X


# dropping columns
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, variables_to_drop=None):
        self.variables_to_drop = variables_to_drop

    def fit(self, X, y=None):
        return self  # Nothing to fit, so just return self

    def transform(self, X):
        X = X.drop(columns=self.variables_to_drop, errors="ignore")
        return X


# imputing mode for categorical features
class ModeImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
        self.mode_dict = {}

    def fit(self, X, y=None):
        for col in self.variables:
            self.mode_dict[col] = X[col].mode()[0]
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.variables:
            X.fillna({col: self.mode_dict}, inplace=True)
        return X


# imputing median for numerical features
class MedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
        self.median_dict = {}

    def fit(self, X, y=None):
        for col in self.variables:
            # Calculate and store the median for each variable
            self.median_dict[col] = X[col].median()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.variables:
            # Fill missing values with the stored median for each variable
            X.fillna({col: self.median_dict}, inplace=True)
        return X


# imputing mean for numerical features
class MeanImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
        self.mean_dict = {}

    def fit(self, X, y=None):
        for col in self.variables:
            self.mean_dict[col] = X[col].mean()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.variables:
            X.fillna({col: self.mean_dict}, inplace=True)
        return X


# replacing non commom values for the variable OTHER
class DataFrameProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, features, quantile_threshold=0.1):
        self.features = features
        self.quantile_threshold = quantile_threshold

    def fit(self, X, y=None):
        self.feature_quantiles_ = {}
        self.feature_value_counts_ = {}

        for feature in self.features:
            value_counts = X[feature].value_counts()
            self.feature_value_counts_[feature] = value_counts
            self.feature_quantiles_[feature] = (
                X[feature].map(value_counts).quantile(self.quantile_threshold)
            )

        return self

    def transform(self, X):
        X = X.copy()  # Make a copy of the input DataFrame to avoid altering original data

        for feature in self.features:
            X[feature + "_count"] = X[feature].map(self.feature_value_counts_[feature])
            quantile_value = self.feature_quantiles_[feature]

            # Apply "OTHER" transformation
            X[feature] = X.apply(
                lambda row: (
                    "OTHER" if X[feature + "_count"][row.name] < quantile_value else row[feature]
                ),
                axis=1,
            )

        return X


# Winsorizing all the numerical features
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_features, limits=[0.025, 0.025]):
        self.numerical_features = numerical_features
        self.limits = limits

    def fit(self, X, y=None):
        # Winsorization does not require fitting as it does not learn from the data.
        return self

    def transform(self, X):
        X = X.copy()  # Create a copy to avoid altering the original data
        for feature in self.numerical_features:
            X[feature + "_winsor"] = winsorize(X[feature], limits=self.limits)
        return X


# Binnaring in order to reduce the number of decimals
class CoordinateBinner(BaseEstimator, TransformerMixin):
    def __init__(self, columns, decimal_places=3):
        # Initialize with the names of the columns and the number of decimal places
        self.columns = columns
        self.decimal_places = decimal_places

    def fit(self, X, y=None):
        # This transformer does not need to learn anything from the data, so fit does nothing
        return self

    def transform(self, X):
        X = X.copy()  # Make a copy of the input DataFrame to avoid changing it in-place
        for column in self.columns:
            if column in X.columns:
                # Round the specified columns to the given number of decimal places
                new_column_name = f"{column}_{self.decimal_places}"
                X[new_column_name] = np.round(X[column], self.decimal_places)
            else:
                # If a column is not found, raise an error
                raise ValueError(f"Column {column} not found in DataFrame")
        return X


# Cleaning the text
class TextProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, column, blacklist=None):
        # Initialize the lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        self.blacklist = blacklist if blacklist is not None else []
        self.column = column  # Specify the column to process

    def fit(self, X, y=None):
        # Download necessary nltk resources during fitting
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)  # For lemma compatibility
        return self

    def transform(self, X):
        X = X.copy()  # Make a copy of the input DataFrame to avoid altering original data
        if self.column in X.columns:
            X[self.column] = X[self.column].apply(self.clean_text)
        else:
            raise ValueError(f"The column {self.column} does not exist in the DataFrame.")
        return X

    def clean_text(self, text):
        if not isinstance(text, str):
            return text  # Avoid processing NaN or None values
        text = text.lower()
        text = self.remove_unwanted_patterns(text)
        words = text.split()
        words = [self.lemmatize_and_singularize(word) for word in words]
        words = [word for word in words if word not in self.blacklist]
        return " ".join(words).strip()

    def remove_unwanted_patterns(self, text):
        patterns_to_remove = [
            r"[^a-z\s]",  # Remove non-alphabetic characters
            r"@[A-Za-z0-9]+",  # Remove mentions
            r"http\S+",  # Remove URLs
            r"#\S+",  # Remove hashtags
            r"[\W_]+",  # Remove non-word characters
            r"\s+",  # Replace multiple spaces with single space
        ]
        for pattern in patterns_to_remove:
            text = re.sub(pattern, " ", text)
        return text

    def lemmatize_and_singularize(self, word):
        singular = wordnet.morphy(word)  # Get the singular form of the word
        return self.lemmatizer.lemmatize(singular if singular else word)


# Applying TFIDT transformation
class TFIDFTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column, max_features=20):
        self.column = column
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=self.max_features)

    def fit(self, X, y=None):
        # Fit the vectorizer to the text data
        if self.column in X.columns:
            self.vectorizer.fit(X[self.column])
        else:
            raise ValueError(f"The specified column '{self.column}' is not in the DataFrame")
        return self

    def transform(self, X):
        # Check if the transformation can be applied
        if self.column not in X.columns:
            raise ValueError(f"The specified column '{self.column}' is not in the DataFrame")

        # Transform the text data to a TF-IDF representation
        tfidf_features = self.vectorizer.transform(X[self.column])

        # Extract feature names and create a DataFrame for the TF-IDF features
        feature_names = self.vectorizer.get_feature_names_out()
        tfidf_df = pd.DataFrame(tfidf_features.toarray(), index=X.index, columns=feature_names)

        # Concatenate the original DataFrame with the new TF-IDF features
        X = pd.concat([X, tfidf_df], axis=1)

        return X


# Creation of new features
class FeatureToolsProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, max_depth=1, trans_primitives=["add_numeric", "multiply_numeric"]):
        self.max_depth = max_depth
        self.trans_primitives = trans_primitives
        self.es = None
        self.feature_matrix = None
        self.feature_defs = None

    def fit(self, X, y=None):
        # Make a temporary index if none exists
        temp_index_added = False
        if "index" not in X.columns:
            X = X.reset_index(drop=True)
            X["temp_index"] = X.index
            index_name = "temp_index"
            temp_index_added = True
        else:
            index_name = "index"

        # Initialize and add a DataFrame to the EntitySet
        self.es = ft.EntitySet(id="data")
        self.es = self.es.add_dataframe(
            dataframe_name="data",
            dataframe=X,
            index=index_name,
            make_index=False,
        )

        # Run deep feature synthesis
        self.feature_matrix, self.feature_defs = ft.dfs(
            entityset=self.es,
            target_dataframe_name="data",
            max_depth=self.max_depth,
            trans_primitives=self.trans_primitives,
            verbose=True,
        )

        # Remove the temporary index from the feature matrix if it was added
        if temp_index_added:
            self.feature_matrix.reset_index(drop=True, inplace=True)

        return self

    def transform(self, X):
        return self.feature_matrix.copy() if self.feature_matrix is not None else X


# Considering the numerical features of the huge dataset
class NumericalFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, remove_columns, include_datetime=False):
        self.remove_columns = remove_columns
        self.include_datetime = include_datetime

    def fit(self, X, y=None):
        # Select columns with numeric datatypes (includes integer and float types)
        self.categorical_columns_ = X.select_dtypes(
            include=["category", "object", "bool"]
        ).columns.tolist()
        self.numerical_columns_ = X.select_dtypes(include=[np.number]).columns.tolist()

        def remove_element(elements, value):
            if value in elements:
                elements.remove(value)
            else:
                elements
            return elements

        for remove_column in self.remove_columns:
            remove_element(self.numerical_columns_, remove_column)

        # Optionally include datetime columns as numerical features
        if self.include_datetime:
            datetime_cols = X.select_dtypes(include=[np.datetime64]).columns.tolist()
            self.numerical_columns_ += datetime_cols

        return self

    def transform(self, X):
        # Return only the selected numerical columns
        return X[self.categorical_columns_ + self.numerical_columns_]


# Log transformation for numerical features
class LogTransforms(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        # Infer numerical columns if variables are not explicitly provided
        if self.variables is None:
            self.variables = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
        return self

    def transform(self, X):
        # Check if the input is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        X = X.copy()
        for col in self.variables:
            if col in X.columns:
                # Apply logarithmic transformation ensuring no log zero issues
                X[col] = np.log(np.abs(X[col]) + 1)
            else:
                raise ValueError(f"The column {col} is not in the DataFrame")
        return X


# Scaling the data
class DataScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Initialize scalers; no need to pass dataframe or features in __init__
        self.std_scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()
        self.numerical_columns = None  # To store the names of the numerical columns

    def fit(self, X, y=None):
        # Identify numerical columns (assume columns with dtype float are numerical)
        self.numerical_columns = X.select_dtypes(include=["float64", "int64"]).columns
        # Fit the StandardScaler only on the numerical columns
        if self.numerical_columns.size > 0:
            self.std_scaler.fit(X[self.numerical_columns])
            # Transform data and fit MinMaxScaler on transformed data
            X_transformed = self.std_scaler.transform(X[self.numerical_columns])
            self.min_max_scaler.fit(X_transformed)
        return self  # fit should always return self

    def transform(self, X):
        # Check if there are numerical columns to transform
        if self.numerical_columns.size > 0:
            # Apply StandardScaler
            scaled_standard = self.std_scaler.transform(X[self.numerical_columns])
            # Apply MinMaxScaler to the output of StandardScaler
            scaled_min_max = self.min_max_scaler.transform(scaled_standard)
            # Replace the numerical columns with their scaled versions
            X = X.copy()
            X[self.numerical_columns] = scaled_min_max

        return X


# finding the correlations
class CorrelationMatrixProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_features=None, threshold=0.7):
        self.numerical_features = numerical_features
        self.threshold = threshold
        self.features_to_drop_ = []

    def fit(self, X, y=None):
        # Ensure that numerical_features is set, if not use all numeric columns
        if self.numerical_features is None:
            self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

        # Compute the absolute correlation matrix
        corr_matrix = X[self.numerical_features].corr().abs()

        # Determine which features to drop based on the threshold
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.features_to_drop_ = [
            column for column in upper.columns if any(upper[column] > self.threshold)
        ]

        return self

    def transform(self, X):
        # Drop the features identified during fitting
        return X.drop(columns=self.features_to_drop_, errors="ignore")

    def get_sorted_correlations(self):
        s = self.corr_matrix.unstack()
        so = s.sort_values(kind="quicksort", ascending=False)
        return pd.DataFrame(so, columns=["Pearson Correlation"])


# low variance filtering
class FeatureVariance(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_features=None, threshold=0.001):
        self.numerical_features = numerical_features
        self.threshold = threshold

    def fit(self, X, y=None):
        # Handle case where numerical_features is None
        if self.numerical_features is None:
            self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

        # Normalize the numerical features
        normalized_data = normalize(X[self.numerical_features])
        data_scaled = pd.DataFrame(normalized_data, columns=self.numerical_features)

        # Compute variance of each feature
        self.variance_ = data_scaled.var()

        return self

    def transform(self, X):
        # mean_variance = self.variance_.mean()
        significant_vars = [
            var for var in self.variance_.index if self.variance_[var] >= self.threshold
        ]
        self.categorical_columns_ = X.select_dtypes(
            include=["category", "object", "bool"]
        ).columns.tolist()
        # Return only the columns with variance above the mean
        return X[self.categorical_columns_ + significant_vars]

    def get_variance_dataframe(self):
        variance_df = pd.DataFrame(self.variance_, columns=["Variance"])
        variance_df["Column"] = variance_df.index
        variance_df.reset_index(drop=True, inplace=True)
        return variance_df


# transforming the categorical features to numerical
class LabelEncoderProcessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Initialize a dictionary to store encoders for each categorical column
        self.encoders = {}

    def fit(self, X, y=None):
        # Identify and fit Label Encoders for each categorical column
        for column in X.select_dtypes(include=["object", "category"]).columns:
            encoder = LabelEncoder()
            self.encoders[column] = encoder.fit(X[column])  # Handle missing values
        return self

    def transform(self, X):
        # Transform each categorical column using the corresponding Label Encoder
        X = X.copy()  # Avoid changing the original DataFrame
        for column, encoder in self.encoders.items():
            if column in X.columns and column:
                X[column] = encoder.transform(X[column])  # Handle missing values
            else:
                raise ValueError(f"Column '{column}' not found in DataFrame")
        return X

    def fit_transform(self, X, y=None):
        # This method is inherited from TransformerMixin and calls fit() and transform()
        self.fit(X, y)
        return self.transform(X)
