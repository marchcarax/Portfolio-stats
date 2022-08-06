# Data manipulation
import numpy as np
import pandas as pd

# Plotting
import plotly.express as px
import streamlit as st

#Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import stockanalytics
import yfinance as yf
import datetime

def main():

    # Begin Streamlit dashboard
    st.set_page_config(layout="wide")
    st.title('Investment strategies - Performance analysis')
    st.sidebar.caption('Last update: Aug 2022')
    start_date = st.sidebar.date_input(
     "Choose Intial date",
     datetime.date(2019, 1, 1))

    st.markdown('#### Strategy comparison')

    #Portfolio composition and weight
    #if you have international stocks, remember to put the whole yahoo name with the dot
    stocks = ['eng.mc','ele.mc', 'itx.mc', 'bbva.mc', 'vid.mc', 'rep.mc', 'csx', 'nvda', 'ibe.mc', 
             'bac', 'regn', 'msft', 'team', 'fb', 'or.pa', 'atvi', 'gdx', 'spy']
    stocks.sort()
    total_stocks = len(stocks) - 1
    weight = [1/total_stocks]*total_stocks 
 
    #Get data
    df = yf.download(stocks, start = start_date, end = "2022-12-31")
    df = df['Adj Close']
    spy = df.pop('SPY')

    #call cumulative returns
    returns = stockanalytics.cum_returns(df, weight).reset_index()
    returns_spy = (1 + spy.pct_change()[1:]).cumprod().reset_index()
    returns.rename(columns={'Date':'date', 0:'ret'}, inplace=True)

    initial_capital_s1 = st.sidebar.slider('Choose initial capital for strat 1', 10000, 100000, value=50000, step=10000)
    initial_capital_s2_s3 = st.sidebar.slider('Choose initial capital for strat 2&3', 10000, 100000, value=20000, step=10000)
    add_capital = st.sidebar.slider('Choose amount to periodically add', 1000, 5000, value=1000, step=500)
    returns_spy['SPY'] = initial_capital_s1 * returns_spy.SPY

    #Strategy 1: Buy 50000 in 2019 and hold
    returns_s1 = returns.copy()
    returns_s1['ret'] = initial_capital_s1 * returns_s1.ret

    #Strategy 2: Buy every 3 months
    returns_s2 = returns.copy()
    returns_s2 = compute_strat_2(returns_s2, initial_capital_s2_s3, add_capital, start_date)

    #Strategy 3: Buy everytime RSI dips below 40
    # Define our Lookback period (our sliding window)
    window_length = st.sidebar.slider('Choose window lenght for RSI', 10, 60, value=14, step=1)
    returns_s3 = returns.copy()
    returns_s3 = compute_strat_3(returns_s3, initial_capital_s2_s3, add_capital, start_date, window_length)

    #st.dataframe(returns_s3)

    # Call plotly figures
    df_total = returns_s1.copy()
    df_total['benchmark'] = returns_spy.SPY
    df_total['ret_s2'] = returns_s2.ret
    df_total['ret_s3'] = returns_s3.ret
    fig = prepare_full_graph(df_total)
    st.plotly_chart(fig, use_container_width=False)

    risk_free_return = 2.5

    st.markdown('#### Strategy 1: Buy and hold')
    st.markdown('Basic strategy that buys 50K from the period chosen and holds until today')
    mean, stdev = portfolio_info(returns_s1)
    st.write('Portfolio expected annualised return is {} and volatility is {}'.format(mean, stdev))
    st.write('Portfolio sharpe ratio is {}'.format((mean - risk_free_return)/stdev))

    st.markdown('#### Strategy 2: Buy every 3 months')
    st.markdown('After an initial capital investment, we add capital every 3 months')
    mean, stdev = portfolio_info(returns_s2)
    st.write('Portfolio expected annualised return is {} and volatility is {}'.format(mean, stdev))
    st.write('Portfolio sharpe ratio is {}'.format((mean - risk_free_return)/stdev))

    st.markdown('#### Strategy 3: Buy after every month when RSI < 35')
    st.markdown('After an initial capital investment, we add capital every month when RSI is lower than 35 or we wait until that happens')
    mean, stdev = portfolio_info(returns_s3.drop(['rsi', 'buy'], axis=1))
    st.write('Portfolio expected annualised return is {} and volatility is {}'.format(mean, stdev))
    st.write('Portfolio sharpe ratio is {}'.format((mean - risk_free_return)/stdev))

    st.markdown('##### RSI graph')
    fig = px.line(returns_s3, x="date", y='rsi')
    fig.add_hline(y=35, line_color="green", line_dash="dash")
    st.plotly_chart(fig, use_container_width=False)
    st.write('Last RSI data point is {}'.format(returns_s3.rsi[-1:].values))

    st.markdown('##### Buy signals for Strat 3')
    fig = px.line(returns_s3, x="date", y='buy')
    st.plotly_chart(fig, use_container_width=False)

def prepare_full_graph(df):
    return px.line(df, x="date", y=['benchmark', 'ret', 'ret_s2', 'ret_s3'])

def portfolio_info(stocks):

    stocks.drop(['date'], axis=1, inplace=True)
    mean_daily_returns = stocks.pct_change().mean()
    cov_data = stocks.pct_change().cov() 
    portfolio_return = round(np.sum(mean_daily_returns) * 252,2)
    #calculate annualised portfolio volatility
    portfolio_std_dev = round(np.sqrt(cov_data) * np.sqrt(252),2)

    return portfolio_return*100, portfolio_std_dev*100

def compute_strat_2(df, capital, add_capital, start_date):

    date_to_add = start_date + datetime.timedelta(days=90)
    for index, row in df.iterrows():
        if row.date > date_to_add:
            capital += add_capital
            date_to_add += datetime.timedelta(days=90)
            add = True
        df.at[index,'ret'] *= capital

    return df

def compute_strat_3(df, capital, add_capital, start_date, n):

    prices = df.ret
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed >= 0].sum()/n
    down = -seed[seed < 0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1]  # The diff is 1 shorter

        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    df['rsi'] = rsi
    df['buy'] = 0

    date_to_add = start_date + datetime.timedelta(days=30)
    add = False
    for idx, row in df.iterrows():
        if (row.rsi < 35) and (row.date > date_to_add):
            capital += add_capital
            date_to_add = row['date'] + datetime.timedelta(days=30)
            add  = True
        df.at[idx,'ret'] *= capital
        if add: df.at[idx,'buy'] = 1
        add = False

    return df


if __name__=='__main__':
    main()