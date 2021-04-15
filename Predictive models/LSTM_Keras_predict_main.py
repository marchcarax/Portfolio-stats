import os
import LSTM_Keras_model
import pandas as pd

def main():

    #Reads csv file to forecast
    series = pd.read_csv('Predictive models\\Data\\MSFT.csv', header=0, index_col=0)
    #test period:
    n = 20

    model = LSTM_Keras_model.Keras_Model(series, n)
    LSTM_Keras_model.Keras_Forecast(model,series, n)


if __name__ == '__main__':
    #due to incompatibilities between tensorflow and numpy, I downgrade numpy
    try:
        os.system('pip install numpy==1.19.5')
    except Exception:
        pass
    main()