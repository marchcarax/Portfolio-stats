import os
import LSTM_Keras_model
import pandas as pd

def main():

    #Reads csv file and takes the data we need, in this case Adjusted Close prices
    #and divide in train and test db
    series = pd.read_csv('Predictive models\\Data\\MSFT.csv', header=0, index_col=0)
    #test period:
    n = 20

    model = LSTM_Keras_model.Keras_Model(series, n)
    LSTM_Keras_model.Keras_Forecast(model, series, n)


if __name__ == '__main__':
    #I actually want to downgrade numpy to the one compatible with tensorflow
    #its faster just updating tensorflow and the package will uninstall and install the right numpy
    os.system('pip install -U tensorflow')
    main()