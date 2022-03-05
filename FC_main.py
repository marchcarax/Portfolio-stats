from PredictiveModels.montecarlo import montecarlo_model
import pandas as pd

def main():

    series = pd.read_csv('PredictiveModels\\Data\\msft.csv')
    forecast = montecarlo_model(series, 20)
    print(forecast)

    #Arima_Model.main()
    #LSTM_Keras_predict_main.main()

if __name__ == '__main__':
    main()