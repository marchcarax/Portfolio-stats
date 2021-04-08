import pandas as pd
import ffn
import numpy as np
import matplotlib.pyplot as plt
import stockanalytics

def drawdown_plot(stocks: list, df: pd.Series):

    drawdown = df
    # Fill NaN's with previous values
    drawdown = drawdown.fillna(method='ffill')
    # Ignore problems with NaN's in the beginning
    drawdown[np.isnan(drawdown)] = -np.Inf
    
    roll_max = np.maximum.accumulate(drawdown)
    drawdown = drawdown / roll_max - 1.

    plt.figure(figsize = (15,8))
    plt.title('Portfolio Drawdown by stock')
    plt.plot(drawdown)
    plt.legend(stocks)
    plt.show()
    plt.savefig('figures\\Drawdown.png')

def monthly_returns_table(df: pd.Series, wts):
    """
    Display a table containing monthly returns and ytd returns
    for every year in range.
    """
    rets = stockanalytics.cum_returns(df, wts)
    stats = ffn.PerformanceStats(rets)
    print('Table displaying monthly and YTD returns:')
    stats.display_monthly_returns()

def stats_to_csv(df: pd.DataFrame):

    stats = ffn.PerformanceStats(df).to_csv('stats.csv')
