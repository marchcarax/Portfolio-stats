# trading systems 

# Data manipulation
import datetime
import numpy as np
import pandas as pd

# optimizitation
from scipy.optimize import minimize

def compute_strat_2(df: pd.DataFrame, capital: int, add_capital: int, start_date: datetime):
    """Computes simple strategy for buying every 3 months and sells every 150 days"""
    date_to_add = start_date + datetime.timedelta(days=90)
    date_to_take = start_date + datetime.timedelta(days=150)
    for index, row in df.iterrows():
        if row.date > pd.to_datetime(date_to_add):
            capital += add_capital
            date_to_add += datetime.timedelta(days=90)
        if row.date > pd.to_datetime(date_to_take):
            capital *= 0.98  # we take 2% of benefits
            date_to_take += datetime.timedelta(days=150)
        df.at[index, "ret"] *= capital

    return df


def compute_strat_3(df: pd.DataFrame, capital: int, add_capital: int, start_date: datetime, n: int):
    """Computes simple strategy for buying when rsi goes down established threshold"""
    prices = df.ret
    deltas = np.diff(prices)
    seed = deltas[: n + 1]
    up = seed[seed >= 0].sum() / n
    down = -seed[seed < 0].sum() / n
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100.0 - 100.0 / (1.0 + rs)

    for i in range(n, len(prices)):
        delta = deltas[i - 1]  # The diff is 1 shorter

        if delta > 0:
            upval = delta
            downval = 0.0
        else:
            upval = 0.0
            downval = -delta

        up = (up * (n - 1) + upval) / n
        down = (down * (n - 1) + downval) / n

        rs = up / down
        rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    df["rsi"] = rsi
    df["buy"] = 0
    df["sell"] = 0

    date_to_add = start_date + datetime.timedelta(days=30)
    date_to_take = start_date + datetime.timedelta(days=30)
    add = False
    take = False
    for idx, row in df.iterrows():
        if (row.rsi < 35) and (row.date > pd.to_datetime(date_to_add)):
            capital += add_capital
            date_to_add = row["date"] + datetime.timedelta(days=30)
            add = True
        elif row.rsi > 75 and row.date > pd.to_datetime(date_to_take):
            capital *= 0.95
            take = True
            date_to_take = row["date"] + datetime.timedelta(days=30)
        df.at[idx, "ret"] *= capital
        if add:
            df.at[idx, "buy"] = 1
        if take:
            df.at[idx, "sell"] = 1
        add = False
        take = False

    return df

def sharpe_ratio(returns: pd.DataFrame, risk_free_rate: float = 0.001):
    """Calculates sharpe ratio"""
    returns.dropna(inplace=True)
    excess_returns = returns - risk_free_rate
    annualized_excess_returns = excess_returns.mean() * 252
    annualized_volatility = returns.std() * np.sqrt(252)
    sharpe = annualized_excess_returns / annualized_volatility
    return sharpe


def rolling_sharpe(df: pd.DataFrame, n: int = 20, risk_free_rate: float = 0.01):
    """Calculates rolling sharpe ratio"""
    r_sharpe = df.rolling(n).apply(lambda x: sharpe_ratio(x, risk_free_rate))
    return r_sharpe


def compute_strat_4(df: pd.DataFrame, capital: int, add_capital: int, start_date: datetime, buy: int, sell: int):
    """Computes complex strategy for buying/selling when sharpe ratio goes down/up a stablished threshold"""
    df["buy"] = 0
    df["sell"] = 0

    date_to_add = start_date + datetime.timedelta(days=30)
    date_to_take = start_date + datetime.timedelta(days=30)
    add = False
    take = False
    for idx, row in df.iterrows():
        if (row.sharpe < buy) and (row.date > pd.to_datetime(date_to_add)):
            capital += add_capital
            date_to_add = row["date"] + datetime.timedelta(days=30)
            add = True
        if add:
            df.at[idx, "buy"] = 1
        if (row.sharpe > sell) and (row.date > pd.to_datetime(date_to_take)):
            capital *= 0.95  # we take 5% of benefits
            date_to_take = row["date"] + datetime.timedelta(days=90)
            take = True
        if take:
            df.at[idx, "sell"] = 1

        df.at[idx, "ret"] *= capital
        add = False
        take = False

    return df

def compute_strat_5(data: pd.DataFrame, window: int):
    """Computes complex strategy for buying when volatility is low and sell when vol is high"""

    data = calculate_volatility(data, window)

    # Generate buy/sell signals based on volatility comparison
    data['buy'] = np.where(data['volatility'] < data['volatility'].shift(1), 1, 0)
    data['sell'] = np.where(data['volatility'] > data['volatility'].shift(1), 1, 0)
        
    return data

def calculate_volatility(data, window=365):
    # Calculate historical volatility using the window
    data['log_returns'] = np.log(data['ret'] / data['ret'].shift(1))
    data['volatility'] = data['log_returns'].rolling(window=window).std() * np.sqrt(window)
    return data


def compute_strat_6(df: pd.DataFrame, stock1: str, stock2: str, capital: int, window: int):
    """Computes pair trading strategy"""
    ratio, mean_vol = compute_ratios(df, stock1, stock2, window)

    df = df["Adj Close"]

    df["ratio"] = ratio
    df["mean_vol_pct_change"] = mean_vol

    df[stock1] = df[stock1].pct_change()
    df[stock2] = df[stock2].pct_change()

    df["s1_weight"] = 0.5
    df["s2_weight"] = 0.5

    high_interval = np.mean(df.ratio) + 1 * np.std(df.ratio)
    low_interval = np.mean(df.ratio) - 1 * np.std(df.ratio)

    mem = "normal"

    for idx, row in df.iterrows():
        if row.ratio > high_interval:
            df.at[idx, "s1_weight"] = 0.25
            df.at[idx, "s2_weight"] = 0.75
            mem = stock2
        elif row.ratio < low_interval:
            df.at[idx, "s1_weight"] = 0.75
            df.at[idx, "s2_weight"] = 0.25
            mem = stock1
        if mem == stock2:
            df.at[idx, "s1_weight"] = 0.25
            df.at[idx, "s2_weight"] = 0.75
        elif mem == stock1:
            df.at[idx, "s1_weight"] = 0.75
            df.at[idx, "s2_weight"] = 0.25

    df["ret"] = (df[stock1] * df["s1_weight"]) + (df[stock2] * df["s2_weight"])
    df["ret"] = (df["ret"] + 1).cumprod()
    df["ret"] = df["ret"] * capital
    return df

def compute_ratios(df: pd.DataFrame, stock1: str, stock2: str, window: int):
    """Calculates ratios between 2 pair of stocks"""
    df_vol = df["Volume"]
    df_close = df["Adj Close"]
    close_change_1 = df_close[stock1].pct_change()
    close_change_2 = df_close[stock2].pct_change()
    close_change_1 = close_change_1 + 1
    close_change_1 = np.where(close_change_1.isna(), 0, close_change_1)
    close_change_2 = close_change_2 + 1
    close_change_2 = np.where(close_change_2.isna(), 0, close_change_2)

    vol = df_vol[stock2].pct_change()
    vol_arr = np.array(vol + 1)

    mean_vol = mean_filter1d_valid_strided(vol_arr, 90)
    mean_vol = np.insert(mean_vol, 0, np.ones(len(vol_arr) - len(mean_vol)))

    ratio_a = []
    ratio_b = []
    ratio = []
    for i in range(0, len(close_change_1)):
        if i < window:
            ratio_a.append(1)
            ratio_b.append(1)
            ratio.append(1)
        else:
            rr_a = np.prod(close_change_1[i - window : i])
            rr_b = np.prod(close_change_2[i - window : i])
            ratio_a.append(rr_a)
            ratio_b.append(rr_b)
            ratio.append(rr_b / rr_a)
    return ratio, mean_vol


def compute_strat_8(df, capital):
    """
    Fibonacci strategy
    This function assumes that the DataFrame df contains columns like "high," "low," and "close" representing price data for each period. 
    It identifies buy signals when the price touches or crosses below Fibonacci levels and sells when the price hits a certain Fibonacci level (in this case, the 1.0 level). 
    You may need to adjust the logic based on your specific trading strategy, exit criteria, or additional conditions.
    """
    df["buy_signal"] = 0
    df["sell_signal"] = 0

    # Define Fibonacci levels (adjust as needed)
    fibonacci_levels = [0.236, 0.382, 0.618, 0.786, 1.0]

    # Calculate price ranges based on Fibonacci levels
    df["price_range"] = df["high"] - df["low"]
    for level in fibonacci_levels:
        df[f"Fib_{level}"] = df["high"] - (df["price_range"] * level)

    # Implement Fibonacci strategy
    in_position = False
    for idx, row in df.iterrows():
        if not in_position:
            for level in fibonacci_levels:
                # Buy signal: Price reaches or goes below Fibonacci level
                if row["low"] <= row[f"Fib_{level}"]:
                    df.at[idx, "buy_signal"] = 1
                    entry_price = row["close"]
                    in_position = True
                    break
        else:
            # Sell signal: Price reaches or goes above 1.0 Fibonacci level (or other exit criteria)
            if row["high"] >= row["Fib_1.0"]:
                df.at[idx, "sell_signal"] = 1
                exit_price = row["close"]
                # Calculate return
                capital *= exit_price / entry_price
                in_position = False

    return df, capital



def compute_strat_9(df, capital, add_capital, w_buy, w_sell):
    """
    Computes Turtle strategy
    EL: Entry long position
    ExL: exit long position
    """

    df["EL"] = df["ret"].rolling(w_buy).max()
    df["ExL"] = df["ret"].rolling(w_sell).min()

    df["buy"] = 0
    df["sell"] = 0

    InTrade_Long = False

    for idx, row in df.iterrows():
        if (row["ret"] >= row["EL"]) and not InTrade_Long:
            capital += add_capital
            df.at[idx, "buy"] = 1
            InTrade_Long = True
        elif (row["ret"] <= row["ExL"]) and InTrade_Long:
            capital *= 0.95
            df.at[idx, "sell"] = 1
            InTrade_Long = False

        df.at[idx, "ret"] *= capital

    return df
