import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from math import sqrt
from sklearn.metrics import mean_squared_error

def Keras_Model(series, n):

    split_point = len(series) - n
    dataset, validation = series[0:split_point], series[split_point:]
    print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
    dataset.to_csv('Predictive models\\Data\\dataset.csv', index=True)
    validation.to_csv('Predictive models\\Data\\validation.csv', index=True)
    
    #We will do the ARIMA tests and model with logreturns
    dataset_train = pd.read_csv('Predictive models\\Data\\dataset.csv', header=0, index_col=0)
    dataset_train = dataset_train.iloc[:,4:5].values
    print(dataset_train)

    #Scaling data
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(dataset_train)

    #Transforms data: LSTMs expect our data to be in a specific format (3D array)
    X_train = []
    y_train = []
    X_eval = []
    y_eval = []
    for i in range(60, len(training_set_scaled)-1):
        X_train.append(training_set_scaled[i-60:i, 0])
        X_eval.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])
        y_eval.append(training_set_scaled[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_eval, y_eval = np.array(X_eval), np.array(y_eval)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_eval = np.reshape(X_eval, (X_eval.shape[0], X_eval.shape[1], 1))

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

    model.add(layers.Dense(units = 1))

    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    model.fit(X_train, y_train, epochs = 100)

    #Read Real data to compare
    dataset_test = pd.read_csv('Predictive models\\Data\\validation.csv')
    real_stock_price = dataset_test.iloc[:, 4:5].values

    #Prepare data to predict stock prices
    #Reshape the dataset as done previously
    #Use MinMaxScaler to inverse transform the data
    series_close = series['Adj Close']
    inputs = series_close[split_point - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    X_test = []

    for i in range(60, 60+n):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    print(predicted_stock_price)

    #Model error
    rmse = sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
    print('Mean square error between train model and test data is: %.2f'%(rmse))
    #Model evaluation
    score = model.evaluate(X_eval, y_eval, verbose = 0) 
    print('Test loss: %.2f'%(score))
    print('Test accuracy: %.2f'%((1-score)*100))
    
    #Plotting the results
    plt.figure(figsize=(15,8))
    plt.plot(real_stock_price, color = 'red', label = 'Real Data')
    plt.plot(predicted_stock_price, color = 'green', label = 'Predicted Data')
    plt.title('Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('Predictive models\\Prediction graphs\\LSTM_Train_test_prediction.png')
    plt.show()

    #Return model to re-use later in forecast
    return model

def Keras_Forecast(model, series, n):

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

    #Graph results
    plt.figure(figsize=(15,8))
    plt.plot(predicted_stock_price, color = 'green', label = 'Predicted Data')
    plt.title('Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.savefig('Predictive models\\Prediction graphs\\LSTM_prediction.png')
    plt.show()