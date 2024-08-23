

from features.data_preprocessor import preprocess_market_data, drop_na_rows, remove_outliers
from features.feature_engineering import calculate_macd, calculate_moving_average, calculate_roc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.append('..')


def process_multiple_markets(df_per_match, sequence_length=10, val_ratio=0.15, test_ratio=0.1):
    all_sequences_train, all_labels_train = [], []
    all_sequences_val, all_labels_val = [], []
    all_sequences_test, all_labels_test = [], []

    for df in df_per_match:
        # Process each match's DataFrame
        sequences_train, labels_train, sequences_val, labels_val, sequences_test, labels_test = process_single_market(
            df, sequence_length, val_ratio, test_ratio)

        # Append the results to the corresponding lists
        all_sequences_train.append(sequences_train)
        all_labels_train.append(labels_train)
        all_sequences_val.append(sequences_val)
        all_labels_val.append(labels_val)
        all_sequences_test.append(sequences_test)
        all_labels_test.append(labels_test)

    # Combine all matches' data into single arrays
    X_train = np.concatenate(all_sequences_train)
    y_train = np.concatenate(all_labels_train)
    X_val = np.concatenate(all_sequences_val)
    y_val = np.concatenate(all_labels_val)
    X_test = np.concatenate(all_sequences_test)
    y_test = np.concatenate(all_labels_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def process_single_market(df, sequence_length=10, val_ratio=0.15, test_ratio=0.1):
    # Preprocess the dataframe
    processed_df = preprocess_market_data(df)
    processed_cleaned_df = drop_na_rows(processed_df)

    # Calculate synthetic features
    roc = calculate_roc(processed_cleaned_df)
    ma = calculate_moving_average(processed_cleaned_df)
    macd, signal = calculate_macd(processed_cleaned_df)
    processed_cleaned_df = remove_outliers(processed_cleaned_df)

    # Combine the features into a single dataframe
    features_df = pd.concat([processed_cleaned_df,
                             roc.add_suffix('_roc'),
                             ma.add_suffix('_ma'),
                             macd.add_suffix('_macd'),
                             signal.add_suffix('_signal')], axis=1)
    features_df = drop_na_rows(features_df)

    # Normalization (optional)
    scaler = MinMaxScaler()
    features_df = pd.DataFrame(scaler.fit_transform(features_df),
                               index=features_df.index,
                               columns=features_df.columns)

    # Create sequences for LSTM
    sequences, labels = create_sequences(features_df, sequence_length)

    # Train-Val-Test split
    sequences_train_val, sequences_test, labels_train_val, labels_test = train_test_split(
        sequences, labels, test_size=test_ratio, shuffle=False)

    sequences_train, sequences_val, labels_train, labels_val = train_test_split(
        sequences_train_val, labels_train_val, test_size=val_ratio / (1 - test_ratio), shuffle=False)

    return sequences_train, labels_train, sequences_val, labels_val, sequences_test, labels_test


def create_sequences(data, sequence_length=64):
    sequences = []
    labels = []

    # Loop through the data to create sequences
    for i in range(sequence_length, len(data)):
        # Extract sequences and corresponding labels
        seq = data.iloc[i-sequence_length:i].values
        # Assuming the first 3 columns are the LTPs for the 3 runners
        label = data.iloc[i, :3].values
        sequences.append(seq)
        labels.append(label)

    return np.array(sequences), np.array(labels)


def process_multiple_markets_cnn(df_per_match, sequence_length=10, val_ratio=0.15, test_ratio=0.1):
    all_sequences_train, all_labels_train = [], []
    all_sequences_val, all_labels_val = [], []
    all_sequences_test, all_labels_test = [], []

    for df in df_per_match:
        # Process each match's DataFrame
        sequences_train, labels_train, sequences_val, labels_val, sequences_test, labels_test = process_single_market_cnn(
            df, sequence_length, val_ratio, test_ratio)

        # Append the results to the corresponding lists
        all_sequences_train.append(sequences_train)
        all_labels_train.append(labels_train)
        all_sequences_val.append(sequences_val)
        all_labels_val.append(labels_val)
        all_sequences_test.append(sequences_test)
        all_labels_test.append(labels_test)

    # Combine all matches' data into single arrays
    X_train = np.concatenate(all_sequences_train)
    y_train = np.concatenate(all_labels_train)
    X_val = np.concatenate(all_sequences_val)
    y_val = np.concatenate(all_labels_val)
    X_test = np.concatenate(all_sequences_test)
    y_test = np.concatenate(all_labels_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def process_single_market_cnn(df, sequence_length=10, val_ratio=0.15, test_ratio=0.1):
    # Preprocess the dataframe
    processed_df = preprocess_market_data(df)
    processed_cleaned_df = drop_na_rows(processed_df)

    # Calculate synthetic features
    roc = calculate_roc(processed_cleaned_df)
    ma = calculate_moving_average(processed_cleaned_df)
    macd, signal = calculate_macd(processed_cleaned_df)
    processed_cleaned_df = remove_outliers(processed_cleaned_df)

    # Combine the features into a single dataframe
    features_df = pd.concat([processed_cleaned_df,
                             roc.add_suffix('_roc'),
                             ma.add_suffix('_ma'),
                             macd.add_suffix('_macd'),
                             signal.add_suffix('_signal')], axis=1)
    features_df = drop_na_rows(features_df)

    # Normalization (optional)
    scaler = MinMaxScaler()
    features_df = pd.DataFrame(scaler.fit_transform(features_df),
                               index=features_df.index,
                               columns=features_df.columns)

    # Create sequences for LSTM
    sequences, labels = create_sequences_cnn(features_df, sequence_length)

    # Train-Val-Test split
    sequences_train_val, sequences_test, labels_train_val, labels_test = train_test_split(
        sequences, labels, test_size=test_ratio, shuffle=False)

    sequences_train, sequences_val, labels_train, labels_val = train_test_split(
        sequences_train_val, labels_train_val, test_size=val_ratio / (1 - test_ratio), shuffle=False)

    return sequences_train, labels_train, sequences_val, labels_val, sequences_test, labels_test


def create_sequences_cnn(data, sequence_length=10, roc_threshold=0.01):
    sequences = []
    labels = []

    # Assuming first 3 columns are LTPs for the 3 runners
    ltp_columns = data.columns[:3]

    for i in range(sequence_length, len(data) - 1):
        seq = data.iloc[i-sequence_length:i].values

        # Calculate the rate of change between current LTP and LTP 'sequence_length' steps ago
        roc = (data[ltp_columns].iloc[i] - data[ltp_columns].iloc[i-sequence_length]) / data[ltp_columns].iloc[i-sequence_length]

        # Classify RoC: 1 for positive, 0 for negative or neutral
        label = np.where(roc > roc_threshold, 1, 0)

        sequences.append(seq)
        labels.append(label)

    return np.array(sequences), np.array(labels)



