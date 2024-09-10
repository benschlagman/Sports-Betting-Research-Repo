import pandas as pd
import numpy as np


class DataSet:
    X: pd.DataFrame = None
    y: pd.Series = None


class DataProcessor:
    """
    A class to process and handle football match data, including splitting the data
    into training, testing, and validation sets, and providing unique teams from the dataset.

    Parameters:
    df (DataFrame): The input dataframe containing football match data.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.train_data = None
        self.test_data = None
        self.val_data = None

        self.columns_for_test = ['HomeTeam', 'AwayTeam', 'Date', 'B365H', 'B365A',
                                 'B365D', 'PreHXG', 'PreAXG', 'RoundID', 'Season', 'GameWeek']

    def get_unique_teams(self) -> list[str]:
        """
        Returns a list of all unique teams by combining the unique values of HomeTeam and AwayTeam columns.

        Returns:
        list[str]: A list of unique team names from the dataset.
        """
        home_teams = self.df['HomeTeam'].unique().tolist()
        away_teams = self.df['AwayTeam'].unique().tolist()
        return list(set(home_teams + away_teams))

    def split_data(self, validation=False, train_test_ratio=0.8, test_val_ratio=0.5) -> tuple[DataSet, DataSet, DataSet] | tuple[DataSet, DataSet]:
        """
        Splits the dataset into training and testing sets. Optionally, it can also split a validation set from the test data.

        Parameters:
        validation (bool): Whether to create a validation set or not.
        train_test_ratio (float): Proportion of the data used for training.
        test_val_ratio (float): Proportion of the test data split into validation (if validation is True).

        Returns:
        tuple: A tuple of DataSet objects for training, testing, and optionally validation sets.
        """
        if not validation:
            self.train_data, self.test_data = np.split(
                self.df, [int(train_test_ratio * len(self.df))])
            return self.get_train_data(), self.get_test_data()
        else:
            self.train_data, self.test_data = np.split(
                self.df, [int(train_test_ratio * len(self.df))])
            self.test_data, self.val_data = np.split(
                self.test_data, [int(test_val_ratio * len(self.test_data))])
            return self.get_train_data(), self.get_test_data(), self.get_val_data()

    def split_data_last_n(self, n=10) -> tuple[DataSet, DataSet]:
        """
        Splits the dataset by assigning the last 'n' rows to the test set and the rest to the training set.

        Parameters:
        n (int): The number of rows to assign to the test set.

        Returns:
        tuple: A tuple of DataSet objects for training and testing sets.
        """
        self.train_data = self.df[:-n]
        self.test_data = self.df[-n:]

        return self.get_train_data(), self.get_test_data()

    def get_train_data(self) -> DataSet:
        """
        Retrieves the training data, including the features (X) and labels (y).

        Returns:
        DataSet: An object containing training features and labels.
        """
        train = DataSet()
        train.y = self.train_data['FTR']
        train.X = self.train_data

        return train

    def get_test_data(self) -> DataSet:
        """
        Retrieves the test data, ensuring no data leakage by only selecting relevant columns.

        Returns:
        DataSet: An object containing test features and labels.
        """
        test = DataSet()
        test.y = self.test_data['FTR']
        # Avoid data leakage by selecting only the required test columns
        test.X = self.test_data[self.columns_for_test]
        return test

    def get_val_data(self) -> DataSet:
        """
        Retrieves the validation data, ensuring no data leakage by only selecting relevant columns.

        Returns:
        DataSet: An object containing validation features and labels.
        """
        val = DataSet()
        val.y = self.val_data['FTR']
        # Avoid data leakage by selecting only the required validation columns
        val.X = self.val_data[self.columns_for_test]

        return val
