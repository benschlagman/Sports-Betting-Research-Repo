import pandas as pd

def calculate_roc(data, time_periods=1):
    """
    Calculate Rate of Change (RoC) for the dataframe.

    Parameters:
    data (DataFrame): Input time series data.
    time_periods (int): The number of periods for RoC calculation (default is 1).

    Returns:
    DataFrame: RoC values.
    """
    return data.pct_change(periods=time_periods)


def calculate_moving_average(data, window_size=5):
    """
    Calculate Moving Average (MA) for the dataframe.

    Parameters:
    data (DataFrame): Input time series data.
    window_size (int): The size of the moving window (default is 5).

    Returns:
    DataFrame: Moving average values.
    """
    return data.rolling(window=window_size).mean()


def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate Moving Average Convergence Divergence (MACD) and Signal line.

    Parameters:
    data (DataFrame): Input time series data.
    fast_period (int): Period for fast EMA (default is 12).
    slow_period (int): Period for slow EMA (default is 26).
    signal_period (int): Period for signal line calculation (default is 9).

    Returns:
    Tuple: MACD line and Signal line as DataFrames.
    """
    fast_ema = data.ewm(span=fast_period, adjust=False).mean()
    slow_ema = data.ewm(span=slow_period, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.rolling(window=signal_period).mean()
    return macd_line, signal_line
