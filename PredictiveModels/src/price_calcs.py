import pandas as pd
import numpy as np

def price_to_logreturns(series: pd.DataFrame):
    
    price = series['Adj Close']
    returns = price.pct_change()
    logreturns = np.log1p(returns)
    #fill with zeros and create dataframe
    serieslogres = logreturns.fillna(0)
    serieslogres = pd.DataFrame(serieslogres)
    return  serieslogres

def logreturns_to_price(initial_price, series: pd.Series, price: pd.DataFrame):
    
    seriesnormal = np.expm1(series)
    future_price = [initial_price.values]
    price['change'] = seriesnormal
    future_change = price['change']
    j = 1
    i = 0

    while i < (len(future_change)-1):

        new_price = (future_change[j] + 1)*future_price[i]
        future_price.append(new_price)
        i += 1
        j += 1

    price['predict'] = np.array(future_price).reshape(-1)
    return price