import LSTM_Keras_model
import pandas as pd

def main():

    #Reads csv file to forecast
    series = pd.read_csv('PredictiveModels\\Data\\MSFT.csv', header=0, index_col=0)
    #test period:
    n = 20
    keras_calc(series, n)


def keras_calc(series, n):
    model = LSTM_Keras_model.Keras_Model(series, n)
    LSTM_Keras_model.Keras_Forecast(model,series, n)


if __name__ == '__main__':
    main()