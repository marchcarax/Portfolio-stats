import src.price_calcs
import src.arima_calcs
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

def main():

    #Reads csv file and takes the data we need, in this case Adjusted Close prices
    #and divide in train and test db
    series = pd.read_csv('Predictive models\\Data\\MSFT.csv', header=0, index_col=0)
    #test period:
    n = 20
    split_point = len(series) - n
    dataset, validation = series[0:split_point], series[split_point:]
    print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
    dataset.to_csv('Predictive models\\Data\\dataset.csv', index=True)
    validation.to_csv('Predictive models\\Data\\validation.csv', index=True)
    
    #We will do the ARIMA tests and model with logreturns
    series_train = pd.read_csv('Predictive models\\Data\\dataset.csv', header=0, index_col=0)
    logreturns = src.price_calcs.price_to_logreturns(series_train)
        
    #Check stationarity test
    src.arima_calcs.check_stationarity(logreturns.values)
    #Checks best ARIMA model. Write down the results for the ARIMA Model
    bestarima = src.arima_calcs.autoarima(logreturns.values)

    #Arima model
    mod = sm.tsa.statespace.SARIMAX(logreturns.values,order=(int(bestarima[0]),int(bestarima[1]),int(bestarima[2])))
    results = mod.fit()
    print(results.summary())

    #Print Residual Stats 
    residuals = pd.DataFrame(results.resid)
    src.arima_calcs.arima_graphs(residuals)

    #multi-step out-of-sample prediction for next n steps (days in our case)
    future_days = n * 2
    start_index = len(logreturns.values)-n
    end_index = start_index + future_days - 1
    predict = results.predict(start=start_index, end=end_index)

    #Validation data
    validation = pd.read_csv('Predictive models\\Data\\validation.csv', header=0, index_col=0)

    #Returns logreturns to price data
    predict_test = predict[:-n]
    initial_price = validation['Adj Close'][:-len(validation)+1]
    test_data = src.price_calcs.logreturns_to_price(initial_price, predict_test, validation)
    print(test_data)

    #Predict vs validation data
    src.arima_calcs.prediction_traintest_graph(test_data)
    #Forecast for next n days
    src.arima_calcs.prediction_graph(predict, validation[-n:])


if __name__ == '__main__':
    main()