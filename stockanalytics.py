# Data manipulation
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

import statsmodels.api as sm
from statsmodels import regression
from scipy.stats import norm
from tabulate import tabulate
from pandas_datareader import data as web
from datetime import datetime

import yfinance as yf


def cum_returns_graph(stocks, wts):

  #Plots the cumulative returns of your portfolio
  
  cumulative_ret = cum_returns(stocks, wts)
  fig = plt.figure(figsize=(15,8))
  ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
  ax1.plot(cumulative_ret)
  ax1.set_xlabel('Date')
  ax1.set_ylabel("Cumulative Returns")
  ax1.set_title("Portfolio Cumulative Returns")
  plt.show()
  fig.savefig('figures\\cum_returns.png')

def cum_returns(stocks, wts):

  weighted_returns = (wts * stocks.pct_change()[1:])
  port_ret = weighted_returns.sum(axis=1)
  return (port_ret + 1).cumprod() 
    
def cum_returns_benchmark(stocks, wts, benchmark, start_date, end_date):

  cumulative_ret_df1 = cum_returns(stocks, wts)

  df2 = yf.download(benchmark, start = start_date, end= end_date )
  price_data2 = df2['Adj Close']
  return_df2 = price_data2.pct_change()[1:]
  cumulative_ret_df2 = (return_df2 + 1).cumprod()

  df1 = pd.DataFrame(cumulative_ret_df1)
  df2 = pd.DataFrame(cumulative_ret_df2)
  df = pd.concat([df1,df2], axis=1)
  df = pd.DataFrame(df)
  df.columns = ['portfolio', 'benchmark']
  return df

def graph_comparison(stocks, wts, benchmark, start_date, end_date):
  #Compares Portfolio to benchmark
  df = cum_returns_benchmark(stocks, wts, benchmark, start_date, end_date)
  plt.figure(figsize = (15,8))
  plt.plot(df.portfolio,color='r', label = 'Portfolio')
  plt.plot(df.benchmark,color='g', label = 'SPY')
  plt.title('Portfolio vs SPY')
  plt.legend(loc = 'upper center', bbox_to_anchor= (0.5, -0.05), fancybox = True, shadow = True, ncol = 5)
  plt.savefig('figures\\portfolio vs benchmark.png')
  plt.show()

def sharpe_ratio(stocks, wts):

  price_data = stocks
  ret_data = price_data.pct_change()[1:]
  port_ret = (ret_data * wts).sum(axis = 1)
  #cumulative_ret = (port_ret + 1).cumprod()
  geometric_port_return = np.prod(port_ret + 1) ** (252/port_ret.shape[0]) - 1
  annual_std = np.std(port_ret) * np.sqrt(252)
  port_sharpe_ratio = geometric_port_return / annual_std
  #print("Sharpe ratio : %.2f"%(port_sharpe_ratio))
  return port_sharpe_ratio

def sortino_ratio(returns):

  #Calculates the sortino ratio given a series of returns
  returns = returns.values - 1
  res = returns.mean() / returns[returns < 0].std()
  return res

def portfolio_info(stocks, weights):
   
  price_data = stocks
  price_data.sort_index(inplace=True)
  returns = price_data.pct_change()
  mean_daily_returns = returns.mean()
  cov_matrix = returns.cov() 
  portfolio_return = round(np.sum(mean_daily_returns * weights) * 252,2)
  #calculate annualised portfolio volatility
  weights = np.array(weights)
  portfolio_std_dev = round(np.sqrt(np.dot(weights.T,np.dot(cov_matrix, weights))) * np.sqrt(252),2)
  
  print('---------------------------------')
  print('Portfolio expected annualised return is {} and volatility is {}'.format(portfolio_return*100,portfolio_std_dev*100))
  port_sharpe_ratio = sharpe_ratio(stocks, weights)
  print("Sharpe ratio : %.2f"%(port_sharpe_ratio))
  returns = cum_returns(stocks, weights)
  ret_sortino = sortino_ratio(returns)
  print("Sortino ratio : %.2f"%(ret_sortino))
    
def efficient_frontier(stock_list, start_date, end_date, iterations):

  stock_raw = yf.download(stock_list, start=start_date, end=end_date)
  stock = stock_raw['Close']
  #df = pd.DataFrame(stock)
  #port_ret = stock.sum(axis=1)
  log_ret = np.log(stock/stock.shift(1))
  num_runs = iterations

  all_weights = np.zeros((num_runs,len(stock.columns)))
  ret_arr = np.zeros(num_runs)
  vol_arr = np.zeros(num_runs)
  sharpe_arr = np.zeros(num_runs)

  for ind in range(num_runs):

      # Create Random Weights
      weights = np.array(np.random.random(len(stock_list)))

      # Rebalance Weights
      weights = weights / np.sum(weights)

      # Save Weights
      all_weights[ind,:] = weights

      # Expected Return
      ret_arr[ind] = np.sum((log_ret.mean() * weights) *252)

      # Expected Variance
      vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))

      # Sharpe Ratio
      sharpe_arr[ind] = ret_arr[ind]/vol_arr[ind]

  max_sr_ret = ret_arr[sharpe_arr.argmax()]
  max_sr_vol = vol_arr[sharpe_arr.argmax()]

  print('---------------------------------')
  print('Portfolio efficiency analysis:')
  print('Return with Maximum SR: %.2f'%(max_sr_ret*100))
  print('Volality with Maximum SR: %.2f'%(max_sr_vol*100))
  print('Max Sharpe Ratio: %.2f'%(sharpe_arr.max()))
  allocation = [i * 100 for i in all_weights[sharpe_arr.argmax(),:] ]
  print('Optimized allocation (in %):')
  #print(allocation)
  print_and_plot_portfolio_weights(stock_list, allocation)
  print('---------------------------------')

  plt.figure(figsize=(15,8))
  plt.scatter(vol_arr,ret_arr,c=sharpe_arr,cmap='plasma')
  plt.colorbar(label='Sharpe Ratio')
  plt.xlabel('Volatility')
  plt.ylabel('Return')

  # Add red dot for max SR
  plt.scatter(max_sr_vol,max_sr_ret,c='red',s=50,edgecolors='black')
  plt.savefig('figures\\portfolio_efficiency.png')
  plt.show()

def print_and_plot_portfolio_weights(stock_list: list, weights_dictionary: list) -> None:
  
  symbols = stock_list
  symbol_weights = []
  stock_dict = dict(zip(symbols, weights_dictionary))
  for symbol in symbols:
    symbol_weights = stock_dict.get(symbol)
    print("Symbol: %s, Weight: %.2f" %(symbol, symbol_weights))
    #symbol_weights.append(stock_dict[symbol])
    
def calculate_rolling_sharpe_ratio(price_series: pd.Series,
  n: float=20) -> pd.Series:
  """
  Compute an approximation of the Sharpe ratio on a rolling basis. 
  Intended for use as a preference value.
  """
  rolling_return_series = calculate_return_series(price_series).rolling(n)
  return rolling_return_series.mean() / rolling_return_series.std()

def plot_rolling_sharpe_ratio(stocks, wts, n: float=20):

  df = cum_returns(stocks, wts)
  df = pd.DataFrame(df)
  rolling = calculate_rolling_sharpe_ratio(df, n)
  plt.figure(figsize = (15,8))
  plt.plot(rolling, color='r', label = 'Sharpe Ratio')
  plt.title('Portfolio Rolling Sharpe Ratio')
  plt.axhline(y = 0, color = 'b', linestyle = '--')
  plt.show()
  plt.savefig('figures\\rolling_sharpe_ratio.png')

def calculate_return_series(series: pd.Series) -> pd.Series:
  """
  Calculates the return series of a given time series.
  The first value will always be NaN.
  """
  shifted_series = series.shift(1, axis=0)
  return series / shifted_series - 1

