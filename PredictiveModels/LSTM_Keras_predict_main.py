import os
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
    try:
        os.system('deactivate')
    except Exception:
        pass


if __name__ == '__main__':
    #due to incompatibilities between tensorflow and numpy, I created a virtual env with the compatible versions for tensorflow
    #you will need numpy=1.19.5 or below for the file to run
    os.system("MLprepared\\Scripts\\activate.bat")
    main()