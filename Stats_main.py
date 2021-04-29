#Basic libraries
import os
import stockanalytics
import portfolio_corr
import calcs
import to_pdf
import pandas as pd
from pandas_datareader import data as web

#Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def main():

    #Portfolio composition and weight
    #if you have international stocks, remember to put the whole yahoo name with the dot
    stocks = ['fb','orcl']
    weight = [0.5, 0.5]
    
    #Analysis timetable
    start_date = "2021-01-01"
    end_date = "2021-12-31"
 
    #Get data
    df = web.get_data_yahoo(stocks, start = start_date, end = end_date)
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
    #Creates and saves pdf
    to_pdf.main()


if __name__ == '__main__':
    main()
