import pandas as pd
from category_encoders import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

class DataEncoder:
    def __init__(self, data: pd.DataFrame):
        self.data: pd.DataFrame = data

    def one_hot_encode(self) -> pd.DataFrame:
        ohe = OneHotEncoder(cols=['HT', 'AT'], use_cat_names=True)
        encoded_data = ohe.fit_transform(self.data)
        encoded_data = encoded_data.apply(pd.to_numeric, errors='coerce')
        self.encoded_data = encoded_data.reindex(sorted(encoded_data.columns), axis=1)

    def normalize(self):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(self.encoded_data)
        self.scaled_data = pd.DataFrame(scaled_data, columns=self.encoded_data.columns)

    def process(self):
        self.one_hot_encode()
        self.normalize()
        return self.scaled_data


class TargetMapper:
    def __init__(self, target: pd.Series):
        self.target: pd.Series = target

    def encode_labels(self) -> pd.Series:
        label_mapping = {'H': 0, 'A': 1, 'D': 2}
        return self.target.map(label_mapping)


class DataAligner:
    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        self.train_data: pd.DataFrame = train_data
        self.test_data: pd.DataFrame = test_data

    def align_columns(self):
        train_only_cols = set(self.train_data.columns) - set(self.test_data.columns)
        test_only_cols = set(self.test_data.columns) - set(self.train_data.columns)

        for col in train_only_cols:
            self.test_data[col] = 0
        for col in test_only_cols:
            self.train_data[col] = 0

        self.train_data = self.train_data.reindex(sorted(self.train_data.columns), axis=1)
        self.test_data = self.test_data.reindex(sorted(self.test_data.columns), axis=1)
        
        return self.train_data, self.test_data
