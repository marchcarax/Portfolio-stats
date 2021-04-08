import pandas as pd
import yfinance as yf
from finvizfinance.quote import finvizfinance

def fundamental_data(ticker):

    stock_info = finvizfinance(ticker)
    data = {}
    data = stock_info.TickerFundament()
    print(data['Company'])
    print(data['Industry'])
    print('P/E: ' + data['P/E'])
    print('EPS: ' + data['EPS (ttm)'])
    print('S/O: ' + data['Shs Outstand'])
    print('Beta: ' + data['Beta'])

    '''
    Lets define if the stock is in a positive trend or not
    to do that, we will use SMA50 and SMA200
    if both SMA are positive, we will say the stock is in a Strong positive trend
    if SMA200 is positive but SMA50 negative, its in a Weak positive trend
    if both SMA are negative, Strong negative trend
    finally, if SMA200 negative but SMA50 positive, Weak negative trend
    There are better ways to define trend, but this will be a useful quick way to do it

    '''

    sma200 = str(data['SMA200'])
    sma50 = str(data['SMA50'])
    print('---------------------------')
    print('SMA200: ' + sma200)
    print('SMA50: ' + sma50)

    if sma200.startswith('-'):
        if sma50.startswith('-'):
            print(ticker + ' is in a Strong negative trend')
        else:
            print(ticker + ' is in a Weak negative trend')
    else:
        if sma50.startswith('-'):
            print(ticker + ' is in a Weak positive trend')
        else:
            print(ticker + ' is in a Strong positive trend') 
    
    #Print Earnings table
    stock = yf.Ticker(ticker)
    cash_flows = stock.cashflow
    cash_flows = cash_flows/1000000
    df = stock.earnings
    df = df/1000000
    df['Growth (in %)'] = df['Earnings'].pct_change()*100
    sort_df = df.sort_values(by = 'Year', ascending=False)
    print('---------------------------')
    print('Stock earnings in $M:')
    print(sort_df)
    print('---------------------------')

    #Lets do a Free Cash Flow calculation to see how healthy the company is
    print('Free Cash Flow per year:')
    ndf = pd.DataFrame()
    ndf['cfo'] = cash_flows.loc['Total Cash From Operating Activities']
    ndf['capex'] = cash_flows.loc['Capital Expenditures']
    ndf['fcf'] = ndf['cfo'] + ndf['capex']
    print(ndf)