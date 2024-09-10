from features.data_preprocessor import preprocess_market_data, drop_na_rows, remove_outliers
from features.feature_engineering import calculate_macd, calculate_moving_average, calculate_roc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.append('..')

def process_multiple_markets(match_data_list, sequence_length=10, val_ratio=0.15, test_ratio=0.1):
    """
    Processes multiple matches' data, creating training, validation, and test sets for each match.

    Parameters:
    match_data_list (list): List of DataFrames, each containing data for a match.
    sequence_length (int): Length of the sequences for LSTM.
    val_ratio (float): Ratio of validation data.
    test_ratio (float): Ratio of test data.

    Returns:
    tuple: Processed data arrays for training, validation, and testing.
    """
    all_sequences_train, all_labels_train = [], []
    all_sequences_val, all_labels_val = [], []
    all_sequences_test, all_labels_test = [], []

    # Process each match's DataFrame
    for match_df in match_data_list:
        sequences_train, labels_train, sequences_val, labels_val, sequences_test, labels_test = process_single_market(
            match_df, sequence_length, val_ratio, test_ratio)

        # Append the results to the corresponding lists
        all_sequences_train.append(sequences_train)
        all_labels_train.append(labels_train)
        all_sequences_val.append(sequences_val)
        all_labels_val.append(labels_val)
        all_sequences_test.append(sequences_test)
        all_labels_test.append(labels_test)

    # Combine data from all matches into single arrays
    X_train = np.concatenate(all_sequences_train)
    y_train = np.concatenate(all_labels_train)
    X_val = np.concatenate(all_sequences_val)
    y_val = np.concatenate(all_labels_val)
    X_test = np.concatenate(all_sequences_test)
    y_test = np.concatenate(all_labels_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def process_single_market(match_df, sequence_length=10, val_ratio=0.15, test_ratio=0.1):
    """
    Processes a single match's data to create sequences and perform train-test-validation split.

    Parameters:
    match_df (DataFrame): DataFrame containing market data for a single match.
    sequence_length (int): Length of sequences for LSTM.
    val_ratio (float): Ratio of validation data.
    test_ratio (float): Ratio of test data.

    Returns:
    tuple: Training, validation, and testing sequences and labels.
    """
    # Preprocess the DataFrame
    processed_df = preprocess_market_data(match_df)
    cleaned_df = drop_na_rows(processed_df)

    # Generate synthetic features (ROC, moving averages, MACD, signal line)
    roc = calculate_roc(cleaned_df)
    moving_avg = calculate_moving_average(cleaned_df)
    macd_line, signal_line = calculate_macd(cleaned_df)

    # Remove outliers from the cleaned DataFrame
    cleaned_df = remove_outliers(cleaned_df)

    # Combine the features into a single DataFrame
    features_df = pd.concat([cleaned_df,
                             roc.add_suffix('_roc'),
                             moving_avg.add_suffix('_ma'),
                             macd_line.add_suffix('_macd'),
                             signal_line.add_suffix('_signal')], axis=1)
    
    # Drop any rows with missing values in the combined DataFrame
    features_df = drop_na_rows(features_df)

    # Normalization (scaling all features to the range [0, 1])
    scaler = MinMaxScaler()
    normalized_features_df = pd.DataFrame(scaler.fit_transform(features_df),
                                          index=features_df.index,
                                          columns=features_df.columns)

    # Create sequences for LSTM (input features) and labels
    sequences, labels = create_sequences(normalized_features_df, sequence_length)

    # Split data into train, validation, and test sets
    sequences_train_val, sequences_test, labels_train_val, labels_test = train_test_split(
        sequences, labels, test_size=test_ratio, shuffle=False)

    sequences_train, sequences_val, labels_train, labels_val = train_test_split(
        sequences_train_val, labels_train_val, test_size=val_ratio / (1 - test_ratio), shuffle=False)

    return sequences_train, labels_train, sequences_val, labels_val, sequences_test, labels_test


def create_sequences(data, sequence_length=64):
    """
    Creates sequences and corresponding labels for LSTM.

    Parameters:
    data (DataFrame): The input data.
    sequence_length (int): The length of each sequence.

    Returns:
    tuple: Sequences of data and corresponding labels (LTPs of the 3 runners).
    """
    sequences = []
    labels = []

    # Loop through the data to create sequences
    for i in range(sequence_length, len(data)):
        # Extract sequences and corresponding labels
        sequence_data = data.iloc[i-sequence_length:i].values
        # Assuming the first 3 columns are the LTPs for the 3 runners
        sequence_label = data.iloc[i, :3].values
        sequences.append(sequence_data)
        labels.append(sequence_label)

    return np.array(sequences), np.array(labels)


def process_multiple_markets_cnn(match_data_list, sequence_length=10, val_ratio=0.15, test_ratio=0.1):
    """
    Processes multiple matches' data to create training, validation, and test sets for CNN.

    Parameters:
    match_data_list (list): List of DataFrames, each containing data for a match.
    sequence_length (int): Length of the sequences.
    val_ratio (float): Ratio of validation data.
    test_ratio (float): Ratio of test data.

    Returns:
    tuple: Processed data arrays for training, validation, and testing.
    """
    all_sequences_train, all_labels_train = [], []
    all_sequences_val, all_labels_val = [], []
    all_sequences_test, all_labels_test = [], []

    # Process each match's DataFrame
    for match_df in match_data_list:
        sequences_train, labels_train, sequences_val, labels_val, sequences_test, labels_test = process_single_market_cnn(
            match_df, sequence_length, val_ratio, test_ratio)

        # Append the results to the corresponding lists
        all_sequences_train.append(sequences_train)
        all_labels_train.append(labels_train)
        all_sequences_val.append(sequences_val)
        all_labels_val.append(labels_val)
        all_sequences_test.append(sequences_test)
        all_labels_test.append(labels_test)

    # Combine data from all matches into single arrays
    X_train = np.concatenate(all_sequences_train)
    y_train = np.concatenate(all_labels_train)
    X_val = np.concatenate(all_sequences_val)
    y_val = np.concatenate(all_labels_val)
    X_test = np.concatenate(all_sequences_test)
    y_test = np.concatenate(all_labels_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def process_single_market_cnn(match_df, sequence_length=10, val_ratio=0.15, test_ratio=0.1):
    """
    Processes a single match's data for CNN, performs feature engineering, and splits into train, validation, and test sets.

    Parameters:
    match_df (DataFrame): DataFrame containing market data for a single match.
    sequence_length (int): Length of sequences for CNN.
    val_ratio (float): Ratio of validation data.
    test_ratio (float): Ratio of test data.

    Returns:
    tuple: Training, validation, and testing sequences and labels.
    """
    # Preprocess the DataFrame
    processed_df = preprocess_market_data(match_df)
    cleaned_df = drop_na_rows(processed_df)

    # Generate synthetic features (ROC, moving averages, MACD, signal line)
    roc = calculate_roc(cleaned_df)
    moving_avg = calculate_moving_average(cleaned_df)
    macd_line, signal_line = calculate_macd(cleaned_df)

    # Remove outliers from the cleaned DataFrame
    cleaned_df = remove_outliers(cleaned_df)

    # Combine the features into a single DataFrame
    features_df = pd.concat([cleaned_df,
                             roc.add_suffix('_roc'),
                             moving_avg.add_suffix('_ma'),
                             macd_line.add_suffix('_macd'),
                             signal_line.add_suffix('_signal')], axis=1)
    
    # Drop any rows with missing values
    features_df = drop_na_rows(features_df)

    # Normalize the features using MinMaxScaler
    scaler = MinMaxScaler()
    normalized_features_df = pd.DataFrame(scaler.fit_transform(features_df),
                                          index=features_df.index,
                                          columns=features_df.columns)

    # Create sequences for CNN (input features) and labels
    sequences, labels = create_sequences_cnn(normalized_features_df, sequence_length)

    # Train-Val-Test split
    sequences_train_val, sequences_test, labels_train_val, labels_test = train_test_split(
        sequences, labels, test_size=test_ratio, shuffle=False)

    sequences_train, sequences_val, labels_train, labels_val = train_test_split(
        sequences_train_val, labels_train_val, test_size=val_ratio / (1 - test_ratio), shuffle=False)

    return sequences_train, labels_train, sequences_val, labels_val, sequences_test, labels_test


def create_sequences_cnn(data, sequence_length=10, roc_threshold=0.01):
    """
    Creates sequences and labels for CNN based on the rate of change (ROC).

    Parameters:
    data (DataFrame): The input data.
    sequence_length (int): The length of each sequence.
    roc_threshold (float): Threshold to classify the ROC.

    Returns:
    tuple: Sequences of data and corresponding labels based on ROC.
    """
    sequences = []
    labels = []

    # Assuming first 3 columns are LTPs for the 3 runners
    ltp_columns = data.columns[:3]

    for i in range(sequence_length, len(data) - 1):
        sequence_data = data.iloc[i-sequence_length:i].values

        # Calculate the rate of change between current LTP and LTP 'sequence_length' steps ago
        roc_change = (data[ltp_columns].iloc[i] - data[ltp_columns].iloc[i-sequence_length]) / data[ltp_columns].iloc[i-sequence_length]

        # Classify RoC: 1 for positive, 0 for negative or neutral
        sequence_label = np.where(roc_change > roc_threshold, 1, 0)

        sequences.append(sequence_data)
        labels.append(sequence_label)

    return np.array(sequences), np.array(labels)
