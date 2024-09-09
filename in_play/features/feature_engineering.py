import pandas as pd

def calculate_roc(data, time_periods=1):
    """Calculate Rate of Change (RoC) for the dataframe."""
    return data.pct_change(periods=time_periods)


def calculate_moving_average(data, window_size=5):
    """Calculate Moving Average (MA) for the dataframe."""
    return data.rolling(window=window_size).mean()


def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """Calculate Moving Average Convergence Divergence (MACD) and Signal line."""
    fast_ema = data.ewm(span=fast_period, adjust=False).mean()
    slow_ema = data.ewm(span=slow_period, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.rolling(window=signal_period).mean()
    return macd_line, signal_line