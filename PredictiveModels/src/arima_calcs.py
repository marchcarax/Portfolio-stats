import pandas as pd
import numpy as np
import re
import pmdarima as pm
import src.price_calcs
from math import sqrt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.graphics.tsaplots import plot_pacf
import scipy.stats as stats
from datetime import timedelta

def check_stationarity(series):
    
    #Prints stationarity test, if p-value less than 0.05 we can reject the Null Hypothesis, so we can assume 
    #the Alternate Hypothesis that time series is Stationary seems to be true
    result = adfuller(series,autolag='AIC')
    dfoutput = pd.Series(result[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    print('The test statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('%s: %.3f' % (key, value))

def autoarima(series):
    #Chooses the best value of p,q and d based on the lowest AIC and BIC values
    auto_arima_fit = pm.auto_arima(series, start_p=1, start_q=1,
                             max_p=5, max_q=5, m=12,
                             start_P=0, seasonal=False,
                             d=1, D=1, trace=True,
                             error_action='ignore',
                             suppress_warnings=True,
                             stepwise=True,
                             return_valid_fits=True)
    bestarima = auto_arima_fit[0]
    bestarima = str(bestarima)
    qdp = re.findall('\d+', bestarima)
    #Above code gets the best possible result from Arima to a list with q,d,p
    return qdp[:-4]

def arima_graphs(residuals):
    
    plt.title('Residuals plot')
    plt.plot(residuals)
    plt.show()
    plt.clf()
    
    #Why are these not working?
    arr = np.array(residuals)
    arr = arr.reshape(-1)
    res = pd.Series(arr)
    ax = res.plot.kde()
    plot_normal(arr)
    #plot_pacf(residuals)

def plot_normal(x_range, mu=0, sigma=1, cdf=False, **kwargs):
    '''
    Plots the normal distribution function for a given x range
    If mu and sigma are not provided, standard normal is plotted
    If cdf=True cumulative distribution is plotted
    Passes any keyword arguments to matplotlib plot function
    '''
    x = x_range

    if cdf:
        y = stats.norm.cdf(x, mu, sigma)
    else:
        y = stats.norm.pdf(x, mu, sigma)
    plt.plot(x, y)
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()

def prediction_traintest_graph(series: pd.DataFrame):
    
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    ax.set_title('Train vs Test data ARIMA prediction')
    ax.plot(series['Adj Close'],color='r', label = 'Real price')
    ax.plot(series['predict'],color='g', label = 'Prediction')
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.set_xlabel('Date')
    ax.set_ylabel("Price")
    fig.savefig('PredictiveModels\\Prediction graphs\\Train_test_prediction.png')
    plt.show()

    print(series)
    rmse =sqrt(mean_squared_error(series['Adj Close'], series['predict']))
    print('Mean square error between train model and test data is: %.2f'%(rmse))

def prediction_graph(series_predict, price: pd.DataFrame):

    future_dates = future_date(price)
    df = pd.DataFrame(index = future_dates)
    df['change'] = series_predict
    initial_price = price['Adj Close'][:-len(price)+1]
    df_predict = src.price_calcs.logreturns_to_price(initial_price, series_predict, df)
    df_predict['real'] = price['Adj Close'].ffill()
    df_predict.to_csv('PredictiveModels\\Data\\ARIMA_prediction.csv')

    #Prepare graph
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    ax.set_title('ARIMA prediction')
    ax.plot(df_predict['predict'],color='g', label = 'Prediction')
    ax.plot(df_predict['real'],color='r', label = 'Real Price')
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.set_xlabel('Date')
    ax.set_ylabel("Price")
    plt.axvline(x = len(price)-1, color = 'b')
    fig.savefig('PredictiveModels\\Prediction graphs\\Arima_prediction.png')
    plt.show()  

def future_date(df: pd.DataFrame):
    #it creates the Future dates for the graphs
    date_ori = pd.to_datetime(df.index).tolist()
    for i in range(len(df)):
        date_ori.append(date_ori[-1] + timedelta(days = 1))
    date_ori = pd.Series(date_ori).dt.strftime(date_format = '%Y-%m-%d').tolist()
    return date_ori
