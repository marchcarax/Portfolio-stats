# Data manipulation
import numpy as np
import pandas as pd

# Plotting
import plotly.express as px
import streamlit as st

#Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import datetime

def main():

    # Begin Streamlit dashboard
    st.set_page_config(layout="wide")

    st.title('Portfolio strategy analysis')
    st.write('by [Marc](https://www.linkedin.com/in/marc-hernandez-fernandez-4481528b/)')

    st.markdown('')
    
    st.markdown('The goal of the app is to show how following simple investment strategies can outperform the market in the long term. '
                'It follows the naive principal of having an equal-weighted portfolio and each time you add to the portfolio, you are adding equally to all positions. '
                
    )
    st.sidebar.caption('Last update: Aug 2022')
    start_date = st.sidebar.date_input(
     "Choose Intial date",
     datetime.date(2019, 1, 1))


    #Portfolio composition and weight
    #if you have international stocks, remember to put the whole yahoo name with the dot
    portfolio = st.selectbox('Choose portfolio to analyze', ['My portfolio', 'Other'])
    if portfolio == 'My portfolio':
        stocks = ['eng.mc','ele.mc', 'itx.mc', 'bbva.mc', 'vid.mc', 'rep.mc', 'csx', 'nvda', 'ibe.mc', 
                'bac', 'regn', 'msft', 'team', 'fb', 'or.pa', 'atvi', 'gdx']
    else:
        st.markdown("Put stock tickets separated by commas without spaces")
        sl = st.text_input('Stock list:')
        stocks = sl.split(",")

    stocks.append('spy')
    stocks.sort()
    total_stocks = len(stocks) - 1
    weight = [1/total_stocks]*total_stocks 
 
    #Get data
    @st.cache
    def get_data(stocks, start_date, end_date):
        return yf.download(stocks, start = start_date, end = end_date)
    
    df = get_data(stocks, start_date = start_date, end_date = "2022-12-31")
    df = df['Adj Close']
    spy = df.pop('SPY')
    
    st.markdown('#### Simple strategies comparison')

    #call cumulative returns
    returns = cum_returns(df, weight).reset_index()
    returns_spy = (1 + spy.pct_change()[1:]).cumprod().reset_index()
    returns.rename(columns={'Date':'date', 0:'ret'}, inplace=True)

    initial_capital_s1 = st.sidebar.slider('Choose initial capital for strat 1', 10000, 100000, value=50000, step=10000)
    initial_capital_v2 = st.sidebar.slider('Choose initial capital for other strats', 10000, 100000, value=20000, step=10000)
    add_capital = st.sidebar.slider('Choose amount to periodically add', 1000, 5000, value=1000, step=500)
    returns_spy['SPY'] = initial_capital_s1 * returns_spy.SPY

    #Strategy 1: Buy 50000 in 2019 and hold
    returns_s1 = returns.copy()
    returns_s1['ret'] = initial_capital_s1 * returns_s1.ret

    #Strategy 2: Buy every 3 months
    returns_s2 = returns.copy()
    returns_s2 = compute_strat_2(returns_s2, initial_capital_v2, add_capital, start_date)

    #Strategy 3: Buy everytime RSI dips below 40
    # Define our Lookback period (our sliding window)
    window_length = st.sidebar.slider('Choose window lenght for RSI', 10, 60, value=14, step=1)
    returns_s3 = returns.copy()
    returns_s3 = compute_strat_3(returns_s3, initial_capital_v2, add_capital, start_date, window_length)

    #Strategy 4: Buy everytime rolling sharpe cycles lower
    # Define our Lookback period (our sliding window)
    buy_signal = st.sidebar.slider('Choose buying line for Rolling sharpe ratio', -0.5, 0.0, value=-0.1, step=0.1)
    returns_s4 = returns.copy()
    returns_s4["sharpe"] = rolling_sharpe(returns_s4.ret)
    returns_s4 = compute_strat_4(returns_s4, initial_capital_v2, add_capital, start_date, buy_signal)

    #Strategy 5: Buy whenever there is low volatily and sell at high volatility periods
    returns_s5 = returns.copy()
    returns_s5 = compute_strat_5(returns_s5, returns_spy.set_index('Date'), start_date, initial_capital_v2, add_capital, 20)

    #st.dataframe(returns_s3)

    # Call plotly figures
    df_total = returns_s1.copy()
    df_total['benchmark'] = returns_spy.SPY
    df_total['ret_s2'] = returns_s2.ret
    df_total['ret_s3'] = returns_s3.ret
    df_total['ret_s4'] = returns_s4.ret
    df_total['ret_s5'] = returns_s5.ret

    fig = prepare_full_graph_simple_strats(df_total)
    st.plotly_chart(fig, use_container_width=True)
    st.caption('Benchmark is SPY')

    risk_free_return = 2.5

    st.markdown('#### Strategy 1: Buy and hold')
    st.markdown('Basic strategy that buys 50K from the period chosen and holds until today')
    mean, stdev = portfolio_info(returns_s1)
    st.write('Portfolio expected annualised return is {} and volatility is {}'.format(mean, stdev))
    st.write('Portfolio sharpe ratio is {0:0.2f}'.format((mean - risk_free_return)/stdev))

    st.markdown('#### Strategy 2: Buy every 3 months')
    st.markdown('After an initial capital investment, we add capital every 3 months')
    mean, stdev = portfolio_info(returns_s2)
    st.write('Portfolio expected annualised return is {} and volatility is {}'.format(mean, stdev))
    st.write('Portfolio sharpe ratio is {0:0.2f}'.format((mean - risk_free_return)/stdev))

    st.markdown('#### Strategy 3: Buy after every month when RSI < 35')
    st.markdown('After an initial capital investment, we add capital every month when RSI is lower than 35 or we wait until that happens')
    mean, stdev = portfolio_info(returns_s3.drop(['rsi', 'buy'], axis=1))
    st.write('Portfolio expected annualised return is {} and volatility is {}'.format(mean, stdev))
    st.write('Portfolio sharpe ratio is {0:0.2f}'.format((mean - risk_free_return)/stdev))

    st.markdown('##### RSI graph')
    fig = px.line(returns_s3, x="date", y='rsi')
    fig.add_hline(y=35, line_color="green", line_dash="dash")
    st.plotly_chart(fig, use_container_width=False)
    st.write('Last RSI data point is {}'.format(returns_s3.rsi[-1:].values))

    st.markdown('##### Buy signals for Strat 3')
    fig = px.line(returns_s3, x="date", y='buy')
    st.plotly_chart(fig, use_container_width=False)

    st.markdown('#### Complex strategies comparison')

    fig = prepare_full_graph_complex_strats(df_total)
    st.plotly_chart(fig, use_container_width=True)
    st.caption('Benchmark is SPY')

    st.markdown('#### Strategy 4: Buy everytime rolling sharpe cycles lower')
    st.markdown('After an initial capital investment, we add capital every month when rolling Sharpe ratio cycles lower than 0 and we take capital every 3 months when sharpe ratio higher than 0.6')
    mean, stdev = portfolio_info(returns_s4.drop(['sharpe', 'buy', 'sell'], axis=1))
    st.write('Portfolio expected annualised return is {} and volatility is {}'.format(mean, stdev))
    st.write('Portfolio sharpe ratio is {0:0.2f}'.format((mean - risk_free_return)/stdev))

    st.markdown('##### Rolling sharpe graph')
    fig = px.line(returns_s4, x="date", y='sharpe')
    fig.add_hline(y=buy_signal, line_color="green", line_dash="dash")
    fig.add_hline(y=0.6, line_color="red", line_dash="dash")
    st.plotly_chart(fig, use_container_width=False)
    st.write('Last rolling sharpe data point is {}'.format(returns_s4.sharpe[-1:].values))

    st.markdown('##### Buy & Sell signals for Strat 4')
    fig = px.line(returns_s4, x="date", y=['buy', 'sell'])
    st.plotly_chart(fig, use_container_width=False)

    st.markdown('#### Strategy 5: Buy everytime vol is low and sell when high vol')
    st.markdown('After an initial capital investment, we add capital every month when spy vol is low and we take capital every month when vol is at an extreme')
    mean, stdev = portfolio_info(returns_s5.drop(['buy', 'std'], axis=1))
    st.write('Portfolio expected annualised return is {} and volatility is {}'.format(mean, stdev))
    st.write('Portfolio sharpe ratio is {0:0.2f}'.format((mean - risk_free_return)/stdev))

    st.markdown('##### Volatility graph for SPY')
    fig = px.line(returns_s5, x="date", y='std')
    st.plotly_chart(fig, use_container_width=False)

    st.markdown('##### Buy&Sell signals for Strat 5')
    fig = px.line(returns_s5, x="date", y='buy')
    fig.add_hrect(y0=0, y1=1, line_width=0, fillcolor="green", opacity=0.2)
    fig.add_hrect(y0=0, y1=-1, line_width=0, fillcolor="red", opacity=0.2)
    st.plotly_chart(fig, use_container_width=False)


def prepare_full_graph_simple_strats(df):
    return px.line(df, x="date", y=['benchmark', 'ret', 'ret_s2', 'ret_s3'])

def prepare_full_graph_complex_strats(df):
    return px.line(df, x="date", y=['benchmark', 'ret', 'ret_s4', 'ret_s5'])

def portfolio_info(stocks):

    stocks.drop(['date'], axis=1, inplace=True)
    mean_daily_returns = stocks.pct_change().mean()
    cov_data = stocks.pct_change().cov() 
    portfolio_return = round(np.sum(mean_daily_returns) * 252,2)
    #calculate annualised portfolio volatility
    portfolio_std_dev = round(np.sqrt(cov_data) * np.sqrt(252),2)

    return portfolio_return*100, float(portfolio_std_dev.values)*100

def rolling_sharpe(df, n = 20, risk_free = 0):
    return (df.pct_change().rolling(n).mean() - risk_free) / df.pct_change().rolling(n).std()

def cum_returns(stocks, wts):

  weighted_returns = (wts * stocks.pct_change()[1:])
  port_ret = weighted_returns.sum(axis=1)
  return (port_ret + 1).cumprod() 

def compute_strat_2(df, capital, add_capital, start_date):

    date_to_add = start_date + datetime.timedelta(days=90)
    for index, row in df.iterrows():
        if row.date > date_to_add:
            capital += add_capital
            date_to_add += datetime.timedelta(days=90)
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


def compute_strat_4(df, capital, add_capital, start_date, buy):

    df['buy'] = 0
    df['sell'] = 0
    sell = 0.6

    date_to_add = start_date + datetime.timedelta(days=30)
    date_to_take = start_date + datetime.timedelta(days=30)
    add = False
    take = False
    for idx, row in df.iterrows():
        if (row.sharpe < buy) and (row.date > date_to_add):
            capital += add_capital
            date_to_add = row['date'] + datetime.timedelta(days=30)
            add  = True
        if add: df.at[idx,'buy'] = 1
        if (row.sharpe > sell) and (row.date > date_to_take):
            capital *= 0.95 #we take 2% of benefits
            date_to_take = row['date'] + datetime.timedelta(days=90)
            take = True
        if take: df.at[idx,'sell'] = 1

        df.at[idx,'ret'] *= capital
        add = False
        take = False

    return df

def compute_strat_5(df, spy, start_date, capital, add_capital, window):
    date_to_add = start_date + datetime.timedelta(days=30)
    date_to_take = start_date + datetime.timedelta(days=30)
    
    spy = compute_spy_vol((spy.pct_change()), window)

    df['buy'] = spy['buy'].values
    df['std'] = spy['std'].values

    for idx, row in df.iterrows():
        if (row.buy == 1) and (row.date > date_to_add): 
            capital += add_capital            
            date_to_add = row['date'] + datetime.timedelta(days=30)
        elif (row.buy == -1) and (row.date > date_to_take):
            capital *= 0.98
            date_to_take = row['date'] + datetime.timedelta(days=30)
        df.at[idx,'ret'] *= capital
    return df

def compute_spy_vol(df, window):
    df['std'], std_avg = compute_rolling_std(df, window)
    df['buy'] = 0
    #print(df['std'].values)  

    for idx, row in df.iterrows():
        if row['std']*1.5 > std_avg: df.at[idx,'buy'] = -1
        elif row['std']*0.8 < std_avg: df.at[idx,'buy'] = 1
        else: df.at[idx,'buy'] = 0  
    return df

def compute_rolling_std(df, window):
    #Generate rolling comparison between pairs:
    std_1 = [] #dax
    i = 0
    while i < len(df):
        if i < window: std_1.append(0)
        else:
            std_1.append(np.std(df.SPY[i-window:i]))
        i += 1
    return std_1, np.mean(np.array(std_1))



if __name__=='__main__':
    main()