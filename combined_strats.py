"""
Different strategies that mix a set of strategies 
"""

import datetime
import systems
import pandas as pd

# basic combiation of systems

def basic_combination(df, capital, add_capital, start_date):
    df["buy"] = 0
    df["sell"] = 0
    date_to_take = start_date + datetime.timedelta(days=21)
    date_to_add = start_date + datetime.timedelta(days=7*6)
    
    returns = []
    
    for idx, row in df.iterrows():
        # Signal generation
        buy_signal = (row["buy_s4"] == 1) or \
                     (row["buy_s9"] == 1) or \
                     (row["buy_s5"] == 1) or \
                     (row["buy_s3"] == 1)
                     
        sell_signal = (row["sell_s4"] == 1) or \
                      (row["sell_s9"] == 1) or \
                      (row["sell_s5"] == 1) or \
                      (row["sell_s3"] == 1)
        
        # Capital adjustments
        if buy_signal and row["date"] > pd.to_datetime(date_to_add):
            capital += add_capital
            df.at[idx, "buy"] = 1
            date_to_add = row["date"] + datetime.timedelta(days=21)
        if sell_signal and row["date"] > pd.to_datetime(date_to_take):
            capital *= 0.95
            df.at[idx, "sell"] = 1
            date_to_take = row["date"] + datetime.timedelta(days=7*6)
        
        # Calculate returns
        returns.append(capital)
    
    df["ret"] *= returns

    return df

# voting system
# same as above, but add signals

def voting_system(df, capital):
    df["buy"] = 0
    df["sell"] = 0
    date_to_take = start_date + datetime.timedelta(days=21)
    date_to_add = start_date + datetime.timedelta(days=21)
    
    returns = []

    # Implement super strategy actions based on voting results
    for idx, row in df.iterrows():

        buy_signal = row["buy_s4"] + row["buy_s9"] + row["buy_s3"] + row["buy_s5"]
                     
        sell_signal = row["sell_s4"] + row["sell_s9"] + row["sell_s3"] + row["sell_s5"]

        if buy_signal > 2 and row["date"] > pd.to_datetime(date_to_add):
            capital += add_capital
            df.at[idx, "buy"] = 1
            date_to_add = row["date"] + datetime.timedelta(days=21)
            
        elif sell_signal > 2 and row["date"] > pd.to_datetime(date_to_take):
            capital *= 0.95
            df.at[idx, "sell"] = 1
            date_to_take = row["date"] + datetime.timedelta(days=21)

        # Calculate returns
        returns.append(capital)
            
    df["ret"] *= returns
    
    return df

# weihted voting system

def weighted_voting_system(df, capital):
    # Apply Strategy 2: Turtle system
    df = compute_strat_9(df.copy(), capital, 1000, 20, 10)
    
    # Apply Strategy 3: Rolling Sharpe
    rolling_sharpe_df = rolling_sharpe(df['returns'].to_frame(), n=20)
    df['sharpe'] = rolling_sharpe_df.squeeze()
    df = compute_strat_4(df.copy(), capital, 1000, df['date'].iloc[0], 1, 2)
    
    # Apply Strategy 4: RSI-based strategy
    df = compute_RSI_signals(df.copy())

    # Apply Strategy 1: Fibonacci strategy
    df, capital = compute_strat_8(df.copy(), capital)
    
    # Weighted strategy voting system
    df['weighted_buy_signal'] = (0.4 * df['EL']) + (0.3 * df['sell']) + (0.15 * df['RSI_buy']) + (0.15 * df['buy_signal'])
    df['weighted_sell_signal'] = (0.4 * df['ExL']) + (0.3 * df['buy']) + (0.15 * df['RSI_sell']) + (0.15 * df['sell_signal'])
    
    for idx, row in df.iterrows():
        if row['weighted_sell_signal'] > row['weighted_buy_signal']:
            # Execute sell action based on weighted strategy
            # Update capital, portfolio, or perform relevant actions
            True
        elif row['weighted_buy_signal'] > row['weighted_sell_signal']:
            # Execute buy action based on weighted strategy
            # Update capital, portfolio, or perform relevant actions
            False
    
    return df


def correlation_mixing_system(df, capital, add_capital, start_date, buy_level, sell_level, rsi_buy_level, rsi_sell_level):
    """Computes complex strategy for buying/selling when sharpe ratio goes down/up a stablished threshold"""

    # Calculate rolling Sharpe ratio
    df["sharpe"] = rolling_sharpe(df["ret"], n=20)

    # Calculate RSI
    df["RSI"] = pd.Series(df["close"].pct_change().ewm(span=14).mean(), name="RSI")

    # Calculate signals from all 4 strategies
    df["strategy_1_signal"] = compute_strat_8(df.copy(), capital, add_capital)["buy_signal"]
    df["strategy_2_signal"] = compute_strat_9(df.copy(), capital, add_capital, w_buy=buy_level, w_sell=sell_level)["buy_signal"]
    df["strategy_3_signal"] = compute_strat_4(df.copy(), capital, add_capital, start_date, buy_level, sell_level)["buy_signal"]
    df["strategy_4_signal"] = (df["RSI"] < rsi_buy_level).astype(int) - (df["RSI"] > rsi_sell_level).astype(int)

    # Calculate correlation between the signals
    df["correlation"] = df[["strategy_1_signal", "strategy_2_signal", "strategy_3_signal", "strategy_4_signal"]].corr()

    # Identify periods with high correlation
    correlation_threshold = 0.8
    df["high_correlation"] = df["correlation"].apply(lambda x: 1 if x >= correlation_threshold else 0)

    # Combine signals based on correlation
    df["combined_signal"] = df["high_correlation"] * (df["strategy_1_signal"] + df["strategy_2_signal"] + df["strategy_3_signal"] + df["strategy_4_signal"] / 4)

    # Manage position based on combined signal
    df["buy"] = df["combined_signal"]
    df["sell"] = 0

    return df
