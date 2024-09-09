import pandas as pd
import numpy as np


def preprocess_market_data(df):
    # Convert 'pt' to datetime format
    df['publish_time'] = pd.to_datetime(
        df['pt'], errors='coerce', format='ISO8601')

    # Set the datetime as index for resampling
    df.set_index('publish_time', inplace=True)
    df['id'] = df['id'].astype(str)

    # Pivot the DataFrame to ensure each runner's LTP is a separate column
    df_pivot = df.pivot_table(index='publish_time',
                                columns='id',
                              values=['runner1_ltp',
                                      'runner2_ltp', 'runner3_ltp'],
                              aggfunc='last')

    # Resample at a consistent interval (e.g., every 10 seconds)
    df_resampled = df_pivot.resample('10S').ffill()

    return df_resampled


def drop_na_rows(df):
    """Drop rows with NaN values."""
    return df.dropna()

def remove_outliers(df):
    """Remove rows with outlier values using the IQR (Interquartile Range) method."""
    first_quartile = df.quantile(0.25)
    third_quartile = df.quantile(0.75)
    interquartile_range = third_quartile - first_quartile

    lower_limit = first_quartile - 1.5 * interquartile_range
    upper_limit = third_quartile + 1.5 * interquartile_range

    return df[((df >= lower_limit) & (df <= upper_limit)).all(axis=1)]