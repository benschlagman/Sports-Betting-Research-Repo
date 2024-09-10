import pandas as pd
from category_encoders import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

class DataEncoder:
    """
    This class encodes categorical data using one-hot encoding and normalises numerical data.
    """
    def __init__(self, input_data: pd.DataFrame):
        """
        Initialises the encoder with the input DataFrame.
        
        Parameters:
        input_data (DataFrame): The dataset to be encoded and normalised.
        """
        self.input_data: pd.DataFrame = input_data

    def one_hot_encode(self) -> pd.DataFrame:
        """
        Applies one-hot encoding to the home and away team columns.
        
        Returns:
        DataFrame: The one-hot encoded DataFrame.
        """
        ohe_encoder = OneHotEncoder(cols=['HT', 'AT'], use_cat_names=True)
        encoded_df = ohe_encoder.fit_transform(self.input_data)
        # Convert all columns to numeric and reorder the columns alphabetically
        encoded_df = encoded_df.apply(pd.to_numeric, errors='coerce')
        self.encoded_df = encoded_df.reindex(sorted(encoded_df.columns), axis=1)

    def normalise_data(self):
        """
        Normalises the encoded data using MinMaxScaler, scaling values to the range [0, 1].
        
        Returns:
        DataFrame: The normalised DataFrame.
        """
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(self.encoded_df)
        self.normalised_df = pd.DataFrame(scaled_data, columns=self.encoded_df.columns)

    def process(self):
        """
        Orchestrates the encoding and normalisation processes.
        
        Returns:
        DataFrame: The fully processed (encoded and normalised) dataset.
        """
        self.one_hot_encode()
        self.normalise_data()
        return self.normalised_df


class TargetMapper:
    """
    This class is responsible for mapping categorical match outcomes (H, A, D) to numerical labels.
    """
    def __init__(self, target_series: pd.Series):
        """
        Initialises the target mapper with the target series.
        
        Parameters:
        target_series (Series): The target data to be encoded (e.g., match results).
        """
        self.target_series: pd.Series = target_series

    def encode_labels(self) -> pd.Series:
        """
        Maps the match outcomes ('H', 'A', 'D') to numeric values (0, 1, 2).
        
        Returns:
        Series: The target series with numeric labels.
        """
        outcome_mapping = {'H': 0, 'A': 1, 'D': 2}
        return self.target_series.map(outcome_mapping)


class DataAligner:
    """
    This class aligns the columns of the training and testing datasets by ensuring that they contain the same features.
    """
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Initialises the data aligner with the training and testing data.
        
        Parameters:
        train_df (DataFrame): The training dataset.
        test_df (DataFrame): The testing dataset.
        """
        self.train_df: pd.DataFrame = train_df
        self.test_df: pd.DataFrame = test_df

    def align_columns(self):
        """
        Aligns the columns of the training and test datasets, adding missing columns to each DataFrame as necessary.
        New columns are filled with zeros to ensure both datasets have identical structure.
        
        Returns:
        tuple: The aligned training and testing DataFrames.
        """
        train_extra_cols = set(self.train_df.columns) - set(self.test_df.columns)
        test_extra_cols = set(self.test_df.columns) - set(self.train_df.columns)

        # Add missing columns to test set and initialise them with zeros
        for col in train_extra_cols:
            self.test_df[col] = 0
        # Add missing columns to train set and initialise them with zeros
        for col in test_extra_cols:
            self.train_df[col] = 0

        # Reorder columns alphabetically for both datasets
        self.train_df = self.train_df.reindex(sorted(self.train_df.columns), axis=1)
        self.test_df = self.test_df.reindex(sorted(self.test_df.columns), axis=1)
        
        return self.train_df, self.test_df
