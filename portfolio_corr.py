import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import ffn as f

def portfolio_heatmap(stocks, start_date, end_date):
    data = f.get(stocks, start='2021-01-01', end='2021-03-26')
    ret = data.to_log_returns().dropna()
    #Heatmap of the correlation matrix
    corrMatrix = ret.corr()
    sn.heatmap(corrMatrix, annot=True)
    plt.title('Portfolio Heatmap')
    plt.legend([''])
    plt.show()
    plt.savefig('figures\\portfolio_corr_map.png')