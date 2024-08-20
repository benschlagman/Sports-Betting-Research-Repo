import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from features.feature_engineering import calculate_macd, calculate_moving_average, calculate_roc
from featuresdata_preprocessor import preprocess_market_data, drop_na_rows, remove_outliers
from sklearn.preprocessing import MinMaxScaler


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


def create_sequences(data, sequence_length=10):
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
