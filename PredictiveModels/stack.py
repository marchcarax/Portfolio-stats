import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import montecarlo

def main():
    #Read model Fc files
    arima = pd.read_csv('PredictiveModels\\Data\\ARIMA_prediction.csv', header=0, index_col=0)
    real_price = arima.real
    keras_lstm = pd.read_csv('PredictiveModels\\Data\\KerasLSTM_prediction.csv', header=0, index_col=0)
    series = pd.read_csv('PredictiveModels\\Data\\msft.csv')
    n_pred = 10 #predicted days
    montecarlo_res = montecarlo.montecarlo_model(series, n_pred)
    montecarlo_res = np.insert(montecarlo_res, 0, np.zeros(len(real_price)-n_pred))
    stack_predict = np.vstack([arima['predict'].values, keras_lstm['prediction'].values]).T
    
    #Prepare data
    #real_price_scale = np.where(np.isnan(real_price_scale), 0, real_price_scale)
    real_price_monte = np.where(np.isnan(real_price), montecarlo_res, real_price)
    print(real_price_monte)
    minmax = MinMaxScaler().fit(stack_predict.reshape(-1,1))
    stack_predict_scale = MinMaxScaler().fit_transform(stack_predict)
    real_price_scale = MinMaxScaler().fit_transform(np.array(real_price_monte).reshape(-1,1))

    #Additional preparations for logistic regression
    where_below_0 = np.where(stack_predict_scale < 0)
    where_higher_1 = np.where(stack_predict_scale > 1)
    stack_predict_scale[where_below_0[0], where_below_0[1]] = 0
    stack_predict_scale[where_higher_1[0], where_higher_1[1]] = 1

    #xgb (Gradient boost trees) parameters
    params_xgd = {
        'max_depth': 7,
        'objective': 'reg:logistic',
        'learning_rate': 0.05,
        'n_estimators': 10000
    }

    bst = xgb.XGBRegressor(**params_xgd)
    bst.fit(stack_predict_scale,real_price_scale, eval_set=[(stack_predict_scale, real_price_scale)],
             eval_metric='rmse', early_stopping_rounds=20, verbose=False)

    #Gradient boost prediction
    ypred = bst.predict(stack_predict_scale, ntree_limit=bst.best_ntree_limit)
    ypred = minmax.inverse_transform(ypred.reshape(-1,1))
    
    #Plot prediction vs test data
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    ax.plot(arima['predict'].values, color = 'red', label = 'arima')
    ax.plot(keras_lstm['prediction'].values, color = 'yellow', label = 'keras lstm')
    ax.plot(real_price, color = 'black', label = 'real price')
    ax.plot(ypred, color = 'green', label = 'xgboost')
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.set_title('Prediction')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    plt.legend()
    fig.savefig('PredictiveModels\\Prediction graphs\\XGB_Arima_LSTM_comp.png')
    plt.show()

if __name__ == '__main__':
    main()

