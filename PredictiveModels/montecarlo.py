import pandas as pd
import numpy as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def montecarlo_model(series: pd.DataFrame, n: int):

    number_simulation = 100
    predict_day = n

    returns = pd.DataFrame(close).pct_change()
    last_price = close[-1]
    results = pd.DataFrame()
    avg_daily_ret = returns.mean()
    variance = returns.var()
    daily_vol = returns.std()
    daily_drift = avg_daily_ret - (variance / 2)
    drift = daily_drift - 0.5 * daily_vol ** 2

    results = pd.DataFrame()

    for i in tqdm(range(number_simulation)):
        prices = []
        prices.append(aux.iloc[-1])
        for d in range(predict_day):
            shock = [drift + daily_vol * np.random.normal()]
            shock = np.mean(shock)
            price = prices[-1] * np.exp(shock)
            prices.append(price)
        results[i] = prices

    #returns flattened array
    raveled = results.values.ravel() 
    raveled.sort()
    cp_raveled = raveled.copy()

    #Simulations graphs
    plt.figure(figsize=(17,5))
    plt.subplot(1,3,1)
    plt.plot(results)
    plt.ylabel('Value')
    plt.xlabel('Simulated days')
    plt.subplot(1,3,2)
    sns.distplot(aux,norm_hist=True)
    plt.title('$\mu$ = %.2f, $\sigma$ = %.2f'%(aux.mean(),aux.std()))
    plt.subplot(1,3,3)
    sns.distplot(raveled,norm_hist=True,label='monte carlo samples')
    sns.distplot(aux,norm_hist=True,label='real samples')
    plt.title('simulation $\mu$ = %.2f, $\sigma$ = %.2f'%(raveled.mean(),raveled.std()))
    plt.legend()
    plt.show()
    
    return raveled