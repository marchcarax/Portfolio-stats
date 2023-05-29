#Basic libraries
import os
import stockanalytics
import portfolio_corr
import calcs
import to_pdf
import pandas as pd
import numpy as np
from pandas_datareader import data as web
import yfinance as yf
import riskfolio as rp
import matplotlib.pyplot as plt

#Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def main():

    #Portfolio composition and weight
    #if you have international stocks, remember to put the whole yahoo name with the dot
    stocks = ['eng.mc','ele.mc', 'gdx', 'bbva.mc', 'rep.mc', 'csx', 'nvda', 'ibe.mc', 'bac',
             'regn', 'msft', 'rblx', 'fb', 'atvi', 'vid.mc', 'or.pa', 'itx.mc', 'team'] 
    
    #stocks = ['btc', 'eth', 'link']
    #stocks = ['eng.mc','ele.mc','bbva.mc', 'rep.mc','ibe.mc','vid.mc', 'or.pa', 'itx.mc']
    #stocks = ['gdx','csx', 'nvda','bac', 'regn', 'msft', 'rblx', 'fb', 'atvi']
    #stocks = ['googl', 'fb', 'amzn']
    stocks.sort()
    total_stocks = len(stocks)
    weight = [1/total_stocks]*total_stocks
    real_weights = [0.04, 0.07, 0.09, 0.05, 0.07, 0.05, 0.07, 0.08, 0.05, 0.04, 0.06, 0.1, 0.03, 0.06, 0.04, 0.01, 0.05, 0.03]
    linkages = ['ward', 'DBHT']

    real = True
    if real: weight = real_weights
    
    #Analysis timetable
    start_date = "2018-01-01"
    end_date = "2022-12-31"
 
    #Get data
    df = yf.download(stocks, start = start_date, end = end_date)
    df = df['Adj Close']

    #Cumulative Returns
    stockanalytics.cum_returns_graph(df, weight)
    #Portfolio info: gives sharpe ratio and annualised returns and vol
    stockanalytics.portfolio_info(df, weight)

    #Show Drawdown by stocks
    calcs.drawdown_plot(stocks, df)

    #Cumulative Returns vs Benchmark, usually SPY, but can be changed
    stockanalytics.graph_comparison(df, weight, 'SPY', start_date, end_date)

    #Correlation heatmap
    portfolio_corr.portfolio_heatmap(stocks, start_date, end_date)

    #Plots rolling sharpe ratio. it can give a good idea where take some profits (at high extremes)
    #and where add to the portfolio (using portfolio efficiency or other tools to add to the correct stock)
    stockanalytics.plot_rolling_sharpe_ratio(df, weight)

    #Efficient frontier graph with optimized allocation printed in terminal
    stockanalytics.efficient_frontier(stocks, start_date, end_date, 1000)

    #Displays monthly returns and YTD, useful to understand if there is a pattern with strong months and 
    #weak months. Is "sell in may and go away" true to your portfolio?
    calcs.monthly_returns_table(df, weight)

    #Outputs to txt file for easy reading and keeping history performance
    #Takes double time to output the txt, its just faster to copy and paste, feel free to hide the code
    try:
        os.remove('output.txt')
    except Exception:
        pass
    
    os.system('python .\Stats_main.py > output.txt')

    #Create Hierarchical clustering graphs
    df.columns = stocks
    Y = df[stocks].pct_change().dropna()
    fig, ax = plt.subplots(len(linkages), 1, figsize=(12, 15))
    ax = np.ravel(ax)
    j=0
    for i in linkages:
        ax[j] = rp.plot_network(returns=Y,
        codependence='pearson',
        linkage=i,
        k=None,
        max_k=10,
        leaf_order=True,
        kind='spring',
        seed=0,
        ax=ax[j])
        j+=1
        
    plt.savefig('figures\\z_network_graph.png')

    #Creates and saves pdf
    to_pdf.main()


if __name__ == '__main__':
    main()
