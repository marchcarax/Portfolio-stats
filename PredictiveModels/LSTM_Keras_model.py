import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

def Keras_Model(series, n):

    #due to incompatibilities between tensorflow and numpy, I downgrade numpy
    try:
        os.system('pip install numpy==1.19.5 --user')
    except Exception:
        pass

    split_point = len(series) - n
    dataset, validation = series[0:split_point], series[split_point:]
    print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
    dataset.to_csv('PredictiveModels\\Data\\dataset.csv', index=True)
    validation.to_csv('PredictiveModels\\Data\\validation.csv', index=True)
    
    #We will do the ARIMA tests and model with logreturns
    dataset_train = pd.read_csv('PredictiveModels\\Data\\dataset.csv', header=0, index_col=0)
    dataset_train = dataset_train.iloc[:,4:5].values

    #Scaling data
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(dataset_train)

    #Transforms data: LSTMs expect our data to be in a specific format (3D array)
    X_train = []
    y_train = []

    for i in range(60, len(training_set_scaled)-1):
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    #Sequential for initializing the neural network
    #Dense for adding a densely connected neural network layer
    #LSTM for adding the Long Short-Term Memory layer
    #Dropout for adding dropout layers that prevent overfitting
    model = keras.Sequential()

    model.add(layers.Reshape(target_shape = (60,1), input_shape = (X_train.shape[1], 1)))
    model.add(layers.LSTM(units = 50, return_sequences = True))
    model.add(layers.Dropout(0.2))

    model.add(layers.LSTM(units = 50, return_sequences = True))
    model.add(layers.Dropout(0.2))

    model.add(layers.LSTM(units = 50, return_sequences = True))
    model.add(layers.Dropout(0.2))

    model.add(layers.LSTM(units = 50))
    model.add(layers.Dropout(0.2))

    #LSTM for time series should have Dense(1)
    model.add(layers.Dense(units = 1))

    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    model.fit(X_train, y_train, epochs = 100)
    model.summary()
    #Read Real data to compare
    dataset_test = pd.read_csv('PredictiveModels\\Data\\validation.csv')
    real_stock_price = dataset_test.iloc[:, 4:5].values
    dataset_test.set_index('Date', inplace = True)
    print(dataset_test)

    #Prepare data to predict stock prices
    #Reshape the dataset as done previously
    #Use MinMaxScaler to inverse transform the data
    series_close = series['Adj Close']
    inputs = series_close[split_point - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    y_pred = []

    for i in range(60, 60+n):
        y_pred.append(inputs[i-60:i, 0])
    y_pred = np.array(y_pred)
    y_pred = np.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1], 1))
    y_pred = model.predict(y_pred)
    predicted_stock_price = sc.inverse_transform(y_pred)

    #Model error
    rmse = sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
    print('Mean square error between train model and test data is: %.2f'%(rmse))
    
    #Plotting the results
    plt.figure(figsize=(15,8))
    plt.plot(predicted_stock_price, color = 'green', label = 'Predicted Data')
    plt.plot(dataset_test['Adj Close'], color = 'red', label = 'Real Data')
    plt.xticks(dataset_test.index)
    plt.title('Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('PredictiveModels\\Prediction graphs\\LSTM_Train_test_prediction.png')
    plt.show()

    #Save csv data for test vs train LSTM keras
    dataset_test['lstm_pred'] = predicted_stock_price
    dataset_test.to_csv('PredictiveModels\\Data\\KerasLSTM_prediction.csv')

    #Return model to re-use later in forecast
    return model

def Keras_Forecast(model, series, n):

    #Prepare Future Dates
    future_dates = future_date(series.iloc[-n:,:])
    df = pd.DataFrame(index = future_dates)
    df = df[-n:]

    #Prepare data
    series_close = series.iloc[:,4:5].values
    sc = MinMaxScaler(feature_range = (0, 1))
    data_scaled = sc.fit_transform(series_close)

    X_train = []
    y_train = []

    for i in range(60, len(data_scaled)-1):
        X_train.append(data_scaled[i-60:i, 0])
        y_train.append(data_scaled[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    #Fit model with data to predict
    model.fit(X_train, y_train, epochs = 100)

    #Creates prediction given a pred_len
    pred_seq = []
    pred_len = n
    predicted = []

    #Last 60 prices
    current = X_train[len(X_train)-1]
    k=0

    for i in range(0, pred_len):
        predicted.append(model.predict(current[None, :, :])[0,0])
        current = current[1:]
        #its adding the new element (predictde value) at the end of the array
        current = np.insert(current, len(current), predicted[-1], axis=0)
        
    pred_seq.append(predicted)
    predicted_stock_price = sc.inverse_transform(pred_seq).reshape(-1,1)
    df['estimate'] = predicted_stock_price

    #Graph results
    plt.figure(figsize=(15,8))
    plt.plot(df['estimate'], color = 'green', label = 'Predicted Data')
    plt.title('Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.savefig('PredictiveModels\\Prediction graphs\\LSTM_prediction.png')
    plt.show()

    #Save predicted prices + last n days in test vs train data
    df_predict = pd.read_csv('PredictiveModels\\Data\\KerasLSTM_prediction.csv')
    df_predict.set_index('Date', inplace = True)
    df_predict = pd.concat([df_predict, df])
    df_predict["prediction"] = np.where(df_predict["lstm_pred"].isna(),df_predict["estimate"],df_predict["lstm_pred"]).astype("float")
    df_predict = df_predict.drop("estimate",axis=1)
    df_predict = df_predict.drop("lstm_pred",axis=1)
    print(df_predict)
    df_predict.to_csv('PredictiveModels\\Data\\KerasLSTM_prediction.csv')

def future_date(df: pd.DataFrame):
    #it creates the Future dates for the graphs
    date_ori = pd.to_datetime(df.index).tolist()
    for i in range(len(df)):
        date_ori.append(date_ori[-1] + timedelta(days = 1))
    date_ori = pd.Series(date_ori).dt.strftime(date_format = '%Y-%m-%d').tolist()
    return date_ori