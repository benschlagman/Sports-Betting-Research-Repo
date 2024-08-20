import pandas as pd
import numpy as np


def preprocess_market_data(df):
    # Convert 'pt' to datetime format
    df['publish_time'] = pd.to_datetime(df['pt'])

    # Set the datetime as index for resampling
    df.set_index('publish_time', inplace=True)

    # Pivot the DataFrame to ensure each runner's LTP is a separate column
    df_pivot = df.pivot_table(index='publish_time', 
                              values=['runner1_ltp', 'runner2_ltp', 'runner3_ltp'], 
                              aggfunc='last')

    # Resample at a consistent interval (e.g., every 10 seconds)
    df_resampled = df_pivot.resample('10S').ffill()

    return df_resampled

def drop_na_rows(df):
    """Drop rows with NaN values."""
    return df.dropna()

def remove_outliers(df):
    """Remove rows containing outliers based on the IQR method."""
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[((df >= lower_bound) & (df <= upper_bound)).all(axis=1)]