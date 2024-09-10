import pandas as pd
import numpy as np


def preprocess_market_data(df):
    """
    Preprocesses the market data by converting the 'pt' column to datetime format,
    setting it as the index, and converting the 'id' column to string type. 
    The function then pivots the DataFrame based on the 'id' of the runners and 
    resamples the data every 10 seconds, forward-filling any missing data points.
    
    Parameters:
    df (DataFrame): Input DataFrame containing market data with runner LTP (last traded price) values.
    
    Returns:
    DataFrame: Resampled and pivoted DataFrame with runner prices resampled at 10-second intervals.
    """
    df['publish_time'] = pd.to_datetime(
        df['pt'], errors='coerce', format='ISO8601')

    df.set_index('publish_time', inplace=True)
    df['id'] = df['id'].astype(str)

    df_pivot = df.pivot_table(index='publish_time',
                                columns='id',
                              values=['runner1_ltp',
                                      'runner2_ltp', 'runner3_ltp'],
                              aggfunc='last')

    df_resampled = df_pivot.resample('10S').ffill()

    return df_resampled


def drop_na_rows(df):
    """
    Drop rows with NaN values.

    Parameters:
    df (DataFrame): Input DataFrame that may contain NaN values.

    Returns:
    DataFrame: A DataFrame with all rows containing NaN values removed.
    """
    return df.dropna()

def remove_outliers(df):
    """
    Remove rows with outlier values using the IQR (Interquartile Range) method.
    Parameters:
    df (DataFrame): Input DataFrame where outlier values may exist.

    Returns:
    DataFrame: A DataFrame with rows containing outlier values removed.
    """
    first_quartile = df.quantile(0.25)
    third_quartile = df.quantile(0.75)
    interquartile_range = third_quartile - first_quartile

    lower_limit = first_quartile - 1.5 * interquartile_range
    upper_limit = third_quartile + 1.5 * interquartile_range

    return df[((df >= lower_limit) & (df <= upper_limit)).all(axis=1)]