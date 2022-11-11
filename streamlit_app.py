# Data manipulation
import numpy as np
import pandas as pd

# Plotting
import plotly.express as px
import streamlit as st

import datetime

#Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# import quant finance libraries
import yfinance as yf

def main():

    # Begin Streamlit dashboard
    st.set_page_config(layout="wide")

    st.title('Portfolio strategy analysis')
    st.write('by [Marc](https://www.linkedin.com/in/marc-hernandez-fernandez-4481528b/)')

    st.markdown('')
    
    st.markdown('The goal of the app is to show how following simple investment strategies can outperform the market in the long term. '
                'It follows the naive principal of having an equal-weighted portfolio and each time you add to the portfolio, you are adding equally to all positions. '
                
    )
    st.markdown("You can also use it to simulate the performance of made-up portfolios.")
    st.markdown("My portfolio consist of a mix of US and European stocks and I try to keep it at less than 20 companies. It changes every 3 to 6 months.")
    st.sidebar.caption('Last update: Nov 2022')
    start_date = st.sidebar.date_input(
     "Choose Intial date",
     datetime.date(2019, 1, 1))


    #Portfolio composition and weight
    #if you have international stocks, remember to put the whole yahoo name with the dot
    portfolio = st.selectbox('Choose portfolio to analyze', ['My portfolio', 'Other'])
    if portfolio == 'My portfolio':
        stocks = ['eng.mc','ele.mc', 'itx.mc', 'bbva.mc', 'vid.mc', 'rep.mc', 'ibe.mc', 'or.pa',
                'san.pa', 'azn', 'regn', 'atvi', 'msft', 'team', 'googl', 'nvda', 'csx']
    else:
        st.markdown("Put stock tickets separated by commas without spaces (e.g. qqq,msft,aapl,ibe.mc)")
        sl = st.text_input('Stock list:')
        stocks = sl.split(",")

    if stocks[0] == '':
        st.write("Waiting stock tickets inputs...")
    else:
        stocks.sort()
        total_stocks = len(stocks)
        weight = [1/total_stocks]*total_stocks 
    
        #Get data
        @st.cache(ttl=86000)
        def get_data(stocks, start_date, end_date):
            return yf.download(stocks, start = start_date, end = end_date)
        
        df = get_data(stocks, start_date = start_date, end_date = "2022-12-31")
        df = df['Adj Close']
        spy = get_data('spy', start_date = start_date, end_date = "2022-12-31")
        spy = spy[['Adj Close']]
        spy.rename(columns={'Adj Close':'SPY'}, inplace=True)
        
        st.markdown('#### Simple strategies comparison')

        #call cumulative returns
        returns = cum_returns(df, weight).reset_index()
        returns_spy = (1 + spy.pct_change()[1:]).cumprod().reset_index()
        returns.rename(columns={'Date':'date', 0:'ret'}, inplace=True)

        initial_capital_s1 = st.sidebar.slider('Choose initial capital for B&H', 10000, 100000, value=50000, step=10000)
        initial_capital_v2 = st.sidebar.slider('Choose initial capital for incremental strats', 10000, 100000, value=20000, step=10000)
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

        #Strategy 6: Pairs trading
        # change this strategy to allow any pairs
        # you can re-do this strategy by putting anything vs SPY and analyzing the mean rolling returns vs each other
        # or do a second derivative. research again how it works 
        
        #df_pairs = get_data(['qqq', 'iwm'], start_date = start_date, end_date = "2022-12-31")
        #df_pairs = df_pairs[['Adj Close', 'Volume']]
        #returns_s6 = compute_strat_6(df_pairs, initial_capital_s1, 20)
        #returns_s6.reset_index(inplace=True)
        #returns_s6.rename(columns={'Date':'date', 'ret':'ret_s6'}, inplace=True)
        
        #Strategy 7: RSI & EMA cross over


        #Strategy 8: Risk Parity and using pairs trading methods


        #Strategy 9: Turtle's fast system
        returns_s9 = returns.copy()
        returns_s9 = compute_strat_9(returns_s9, initial_capital_v2, add_capital, 20, 10)

        #Strategy 10: Mix of signals
        returns_s10 = returns.copy()
        returns_s10['rsi'] = returns_s3['rsi'].values
        returns_s10['buy_s4'] = returns_s4['buy'].values
        returns_s10['sell_s4'] = returns_s4['sell'].values
        returns_s10['buy_s9'] = returns_s9['buy'].values
        returns_s10['sell_s9'] = returns_s9['sell'].values
        returns_s10 = compute_strat_10(returns_s10, 10000, add_capital, start_date)

        #st.dataframe(returns_s3)

        # Call plotly figures
        df_total = returns_s1.copy()
        returns_spy.rename(columns={'Date':'date', 'SPY':'benchmark'}, inplace=True)
        df_total = pd.merge(df_total, returns_spy[['benchmark', 'date']], how='left', on='date')
        df_total['ret_s2'] = returns_s2.ret
        df_total['ret_s3'] = returns_s3.ret
        df_total['ret_s4'] = returns_s4.ret
        df_total['ret_s5'] = returns_s5.ret
        #df_total = pd.merge(df_total, returns_s6[['date', 'ret_s6']], how='left', on='date')
        df_total['ret_s9'] = returns_s9.ret
        df_total['ret_s10'] = returns_s10.ret


        fig = prepare_full_graph_simple_strats(df_total)
        st.plotly_chart(fig, use_container_width=True)
        last_date = df_total.date[-1:].values
        st.write('Last price day is {}'.format(last_date[0].astype(object).day))
        st.caption('Benchmark is SPY')

        df_ret = returns_s1.set_index('date')
        df_ret['ret_pct'] = df_ret.ret.pct_change()
        df_ret.drop(['ret'], axis=1, inplace=True)
        df_ret = df_ret.resample('MS').sum()
        df_ret.reset_index(inplace=True)
        df_ret['year'] = df_ret['date'].dt.year

        df_ret_spy = returns_spy.set_index('date')
        df_ret_spy['ret_pct'] = df_ret_spy.benchmark.pct_change()
        df_ret_spy.drop(['benchmark'], axis=1, inplace=True)
        df_ret_spy = df_ret_spy.resample('MS').sum()
        df_ret_spy.reset_index(inplace=True)
        df_ret_spy['year'] = df_ret_spy['date'].dt.year

        month = {
                    1:"Jan",
                    2:"Feb",
                    3:"Mar",
                    4:"Apr",
                    5:"May",
                    6:"Jun",
                    7:"Jul",
                    8:"Aug",
                    9:"Sep",
                    10:"Oct",
                    11:"Nov",
                    12:"Dec"
        }


        df_ret['month'] = df_ret['date'].dt.month
        df_ret_spy['month'] = df_ret_spy['date'].dt.month
        
        df_table = pd.pivot_table(df_ret, values='ret_pct', index=['year'],
                        columns=['month'], aggfunc=np.sum, fill_value=0, sort=False)
        
        df_table_spy = pd.pivot_table(df_ret_spy, values='ret_pct', index=['year'],
                        columns=['month'], aggfunc=np.sum, fill_value=0, sort=False)

        df_table.rename(columns=month, inplace=True)
        df_table_spy.rename(columns=month, inplace=True)

        df_table['YTD'] = df_table.sum(axis=1)
        df_table_spy['YTD'] = df_table_spy.sum(axis=1)

        st.write("Table with monthly returns: ")

        def style_negative(v, props=''):
            return props if v < 0 else None

        st.table(df_table.applymap('{:,.2%}'.format))

        with st.expander("Table with monthly returns vs SPY:"):
            df_rest = df_table - df_table_spy
            st.table(df_rest.applymap('{:,.2%}'.format))
        
        # Adding details section so main screen is less convoluted
        risk_free_return = 2.5

        with st.expander("See detailed data per strategy"):

            st.markdown('#### Strategy 1: Buy and hold')
            st.markdown('Basic strategy that buys 50K from the period chosen and holds until today')
            mean, stdev = portfolio_info(returns_s1)
            st.write('Portfolio expected annualized return is {} and volatility is {}'.format(mean, stdev))
            st.write('Portfolio sharpe ratio is {0:0.2f}'.format((mean - risk_free_return)/stdev))

            st.markdown('#### Strategy 2: Buy every 3 months')
            st.markdown('After an initial capital investment, we add capital every 3 months')
            mean, stdev = portfolio_info(returns_s2)
            st.write('Portfolio expected annualized return is {} and volatility is {}'.format(mean, stdev))
            st.write('Portfolio sharpe ratio is {0:0.2f}'.format((mean - risk_free_return)/stdev))

            st.markdown('#### Strategy 3: Buy after every month when RSI < 35')
            st.markdown('After an initial capital investment, we add capital every month when RSI is lower than 35 or we wait until that happens')
            mean, stdev = portfolio_info(returns_s3.drop(['rsi', 'buy'], axis=1))
            st.write('Portfolio expected annualized return is {} and volatility is {}'.format(mean, stdev))
            st.write('Portfolio sharpe ratio is {0:0.2f}'.format((mean - risk_free_return)/stdev))

            st.markdown('##### RSI graph')
            fig = px.line(returns_s3, x="date", y='rsi')
            fig.add_hline(y=35, line_color="green", line_dash="dash")
            st.plotly_chart(fig, use_container_width=False)
            st.write('Last RSI data point is {}'.format(returns_s3.rsi[-1:].values))

            st.markdown('##### Buy signals for Strat 3')
            fig = px.line(returns_s3, x="date", y='buy')
            st.plotly_chart(fig, use_container_width=False)

            st.markdown('#### Strategy 9: Follow famous Turtle system')
            st.markdown('After an initial capital investment, we add capital every month when price breaks 20 days high and sell every month when price break 10 days low')
            mean, stdev = portfolio_info(returns_s9.drop(['sell', 'buy', 'EL', 'ES', 'ExL', 'ExS'], axis=1))
            st.write('Portfolio expected annualized return is {} and volatility is {}'.format(mean, stdev))
            st.write('Portfolio sharpe ratio is {0:0.2f}'.format((mean - risk_free_return)/stdev))

            st.markdown('##### Buy & Sell signals for Strat 9')
            fig = px.line(returns_s9, x="date", y=['buy', 'sell'])
            st.plotly_chart(fig, use_container_width=False)

        st.markdown('#### Advanced strategies comparison')

        fig = prepare_full_graph_complex_strats(df_total)
        st.plotly_chart(fig, use_container_width=True)
        st.caption('Benchmark is SPY')

        df_ret = returns_s10.set_index('date')
        df_ret['ret_pct'] = df_ret.ret.pct_change()
        df_ret.drop(['ret'], axis=1, inplace=True)
        df_ret = df_ret.resample('MS').sum()
        df_ret.reset_index(inplace=True)
        df_ret['year'] = df_ret['date'].dt.year

        month = {
                    1:"Jan",
                    2:"Feb",
                    3:"Mar",
                    4:"Apr",
                    5:"May",
                    6:"Jun",
                    7:"Jul",
                    8:"Aug",
                    9:"Sep",
                    10:"Oct",
                    11:"Nov",
                    12:"Dec"
        }


        df_ret['month'] = df_ret['date'].dt.month
        
        df_table = pd.pivot_table(df_ret, values='ret_pct', index=['year'],
                        columns=['month'], aggfunc=np.sum, fill_value=0, sort=False)

        df_table.rename(columns=month, inplace=True)

        df_table['YTD'] = df_table.sum(axis=1)

        #st.write("Table with monthly returns if using Strategy 10: ")

        def style_negative(v, props=''):
            return props if v < 0 else None

        #st.table(df_table.applymap('{:,.2%}'.format))


        with st.expander("See detailed data per strategy"):

            st.markdown('#### Strategy 4: Buy everytime rolling sharpe cycles lower')
            st.markdown('After an initial capital investment, we add capital every month when rolling Sharpe ratio cycles lower than 0 and we take capital every 3 months when sharpe ratio higher than 0.6')
            mean, stdev = portfolio_info(returns_s4.drop(['sharpe', 'buy', 'sell'], axis=1))
            st.write('Portfolio expected annualized return is {} and volatility is {}'.format(mean, stdev))
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
            st.write('Portfolio expected annualized return is {} and volatility is {}'.format(mean, stdev))
            st.write('Portfolio sharpe ratio is {0:0.2f}'.format((mean - risk_free_return)/stdev))

            st.markdown('##### Volatility graph for SPY')
            fig = px.line(returns_s5, x="date", y='std')
            st.plotly_chart(fig, use_container_width=False)

            st.markdown('##### Buy&Sell signals for Strat 5')
            fig = px.line(returns_s5, x="date", y='buy')
            fig.add_hrect(y0=0, y1=1, line_width=0, fillcolor="green", opacity=0.2)
            fig.add_hrect(y0=0, y1=-1, line_width=0, fillcolor="red", opacity=0.2)
            st.plotly_chart(fig, use_container_width=False)

                     
            #st.markdown('#### Strategy 6: Using Pairs trading to shifts weights between QQQ and IWM')
            #st.markdown('The main idea of this strat is to keep changing weights between QQQ and IWM depending the strenght of each index')
            #mean, stdev = portfolio_info(returns_s6.drop(['QQQ_weight', 'QQQ', 'IWM_weight', 'IWM', 'ratio', 'mean_vol_pct_change'], axis=1))
            #st.write('Portfolio expected annualized return is {} and volatility is {}'.format(mean, stdev))
            #st.write('Portfolio sharpe ratio is {0:0.2f}'.format((mean - risk_free_return)/stdev))

            #st.markdown('##### Ratio graph for IWM / QQQ')
            #fig = px.line(returns_s6, x="date", y=['ratio', 'mean_vol_pct_change'])
            #st.plotly_chart(fig, use_container_width=False)
            #st.write('Mean volume for last 90 days is {}'.format(returns_s6['mean_vol_pct_change'][-1:].values))
            #st.write('Ratio between QQQ and IWM at {}'.format(returns_s6['ratio'][-1:].values))

            #st.markdown('##### Weights for IWM / QQQ')
            #fig = px.line(returns_s6, x="date", y=['QQQ_weight', 'IWM_weight'])
            #st.plotly_chart(fig, use_container_width=False)
            

            st.markdown('#### Strategy 10: Mix of all other strategies')
            st.markdown('Mixes signals from strategy 2, strategy 4 and strategy 9')
            mean, stdev = portfolio_info(returns_s10[['date', 'ret']])
            st.write('Portfolio expected annualized return is {} and volatility is {}'.format(mean, stdev))
            st.write('Portfolio sharpe ratio is {0:0.2f}'.format((mean - risk_free_return)/stdev))

            st.markdown('##### Buy&Sell signals for Strat 10')
            fig = px.line(returns_s10, x="date", y=['buy', 'sell'])
            st.plotly_chart(fig, use_container_width=False)


def prepare_full_graph_simple_strats(df):
    return px.line(df, x="date", y=['benchmark', 'ret', 'ret_s2', 'ret_s3', 'ret_s9'], color_discrete_sequence=px.colors.qualitative.G10, render_mode="SVG")

def prepare_full_graph_complex_strats(df):
    return px.line(df, x="date", y=['benchmark', 'ret_s4', 'ret_s5', 'ret_s10'], color_discrete_sequence=px.colors.qualitative.G10, render_mode="SVG")

def portfolio_info(stocks):

    stocks.drop(['date'], axis=1, inplace=True)
    mean_daily_returns = stocks.pct_change().mean()
    cov_data = stocks.pct_change().cov() 
    portfolio_return = round(np.sum(mean_daily_returns) * 252,2)
    #calculate annualized portfolio volatility
    portfolio_std_dev = round(np.sqrt(cov_data) * np.sqrt(252),2)

    return portfolio_return*100, float(portfolio_std_dev.values)*100

def rolling_sharpe(df, n = 20, risk_free = 0):
    return (df.pct_change().rolling(n).mean() - risk_free) / df.pct_change().rolling(n).std()

def cum_returns(stocks, wts):

  weighted_returns = (wts * stocks.pct_change()[1:])
  weighted_returns = pd.DataFrame(weighted_returns)
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


    spy.drop(['SPY'], axis=1, inplace=True)
    spy.reset_index(inplace=True)
    spy.rename(columns={'Date':'date'}, inplace=True)

    #df['buy'] = spy['buy'].values
    #df['std'] = spy['std'].values
    df = pd.merge(df, spy, on='date', how='left')

    for idx, row in df.iterrows():
        if (row.buy == 1) and (row.date > date_to_add): 
            capital += add_capital            
            date_to_add = row['date'] + datetime.timedelta(days=30)
        elif (row.buy == -1) and (row.date > date_to_take):
            capital *= 0.98
            date_to_take = row['date'] + datetime.timedelta(days=30)
        df.at[idx,'ret'] *= capital
    return df

def compute_strat_6(df, capital, window):

    ratio, mean_vol =  compute_ratios(df, window)
    
    df = df['Adj Close']

    df['ratio'] = ratio
    df['mean_vol_pct_change'] = mean_vol

    df['IWM'] = df['IWM'].pct_change()
    df['QQQ'] = df['QQQ'].pct_change()

    df['QQQ_weight'] = 0.5
    df['IWM_weight'] = 0.5

    high_interval = np.mean(df.ratio) + 1*np.std(df.ratio)
    low_interval = np.mean(df.ratio) - 1*np.std(df.ratio)

    mem = 'normal'

    for idx, row in df.iterrows():
        if row.ratio > high_interval:
            df.at[idx,'QQQ_weight'] = 0.25
            df.at[idx,'IWM_weight'] = 0.75
            mem = 'IWM'
        elif row.ratio < low_interval:
            df.at[idx,'QQQ_weight'] = 0.75
            df.at[idx,'IWM_weight'] = 0.25
            mem = 'qqq'
        if mem == 'IWM':
            df.at[idx,'QQQ_weight'] = 0.25
            df.at[idx,'IWM_weight'] = 0.75
        elif mem == 'qqq':
            df.at[idx,'QQQ_weight'] = 0.75
            df.at[idx,'IWM_weight'] = 0.25

    df['ret'] = (df['QQQ'] * df['QQQ_weight']) + (df['IWM'] * df['IWM_weight'])
    df['ret'] = (df['ret']+1).cumprod()
    df['ret'] = df['ret'] * capital
    return df

def compute_ratios(df, window):
    df_vol = df['Volume']
    df_close = df['Adj Close']
    close_change_1 = df_close['IWM'].pct_change()
    close_change_2 = df_close['QQQ'].pct_change()
    close_change_1 = close_change_1 + 1
    close_change_1 = np.where(close_change_1.isna(), 0, close_change_1)
    close_change_2 = close_change_2 + 1
    close_change_2 = np.where(close_change_2.isna(), 0, close_change_2)

    vol = df_vol['QQQ'].pct_change()
    vol_arr = np.array(vol + 1)
    
    mean_vol = mean_filter1d_valid_strided(vol_arr, 90)
    mean_vol = np.insert(mean_vol, 0, np.ones(len(vol_arr)-len(mean_vol)))
    
    ratio_a = []
    ratio_b = []
    ratio = []
    for i in range(0, len(close_change_1)):
        if i < window:
            ratio_a.append(1)
            ratio_b.append(1)
            ratio.append(1)
        else:
            rr_a = np.prod(close_change_1[i-window:i])
            rr_b = np.prod(close_change_2[i-window:i])
            ratio_a.append(rr_a)
            ratio_b.append(rr_b)
            ratio.append(rr_b/rr_a)
    return ratio, mean_vol


def compute_strat_9(df, capital, add_capital, w_buy, w_sell):
    '''
    EL: Entry long position
    ExL: exit long position
    ES: Entry short positions
    ExS: exit short position
    '''

    df['EL'] = df['ret'].rolling(w_buy).max()
    df['ES'] = df['ret'].rolling(w_buy).min()
    df['ExL'] = df['ret'].rolling(w_sell).min()
    df['ExS'] = df['ret'].rolling(w_sell).max()

    df['buy'] = 0
    df['sell'] = 0

    InTrade_Long = False
    InTrade_Short = False
    bank = 0

    for idx, row in df.iterrows():
        if (row['ret'] >= row['EL']) and (InTrade_Long == False) and (InTrade_Short == False): 
            capital += add_capital
            df.at[idx,'buy'] = 1
            InTrade_Long = True
        elif (row['ret'] <= row['ExL']) and (InTrade_Long == True) and (InTrade_Short == False):
            capital = capital - add_capital/2
            df.at[idx,'sell'] = 1
            InTrade_Long = False
        elif (row['ret'] <= row['ES']) and (InTrade_Long == False) and (InTrade_Short == False):
            capital = capital * 0.98
            bank = capital * 0.02
            df.at[idx,'sell'] = 1
            InTrade_Short = True
        elif (row['ret'] >= row['ExS']) and (InTrade_Long == False) and (InTrade_Short == True):
            capital += bank
            df.at[idx,'buy'] = 1
            InTrade_Short = False
            bank = 0
        
        df.at[idx,'ret'] *= capital

    return df

def compute_strat_10(df, capital, add_capital, start_date):

    df['buy'] = 0
    df['sell'] = 0
    date_to_take = start_date + datetime.timedelta(days=30)
    date_to_add = start_date + datetime.timedelta(days=30)

    for idx, row in df.iterrows():
        # Sharpe buy and sell signals
        if row['buy_s4'] == 1:
            capital += add_capital
            df.at[idx,'buy'] = 1
        if row['sell_s4'] == 1:
            capital = capital - add_capital*0.75
            df.at[idx,'sell'] = 1

        # Turtle system
        if row['buy_s9'] == 1:
            capital += add_capital
            df.at[idx,'buy'] = 1
        if row['sell_s9'] == 1:
            capital = capital - add_capital*0.75
            df.at[idx,'sell'] = 1

        # RSI system
        if row['rsi'] > 73 and row.date > date_to_take:
            capital = capital - add_capital
            date_to_take = row['date'] + datetime.timedelta(days=30)
            df.at[idx,'sell'] = 1
        if row['rsi'] < 35 and row.date > date_to_add:
            capital += add_capital
            date_to_add = row['date'] + datetime.timedelta(days=30)
            df.at[idx,'buy'] = 1
        df.at[idx,'ret'] *= capital

    return df

#Computes mean given a window W and an array a
def mean_filter1d_valid_strided(a, W):
    return strided_app(a, W, S=1).mean(axis=1)


def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

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
    std_1 = []
    i = 0
    while i < len(df):
        if i < window: std_1.append(0)
        else:
            std_1.append(np.std(df.SPY[i-window:i]))
        i += 1
    return std_1, np.mean(np.array(std_1))

if __name__=='__main__':
    main()