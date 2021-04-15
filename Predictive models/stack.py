import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

def main():
    #Read model Fc files
    arima = pd.read_csv('Predictive models\\Data\\ARIMA_prediction.csv', header=0, index_col=0)
    real_price = arima.real
    keras_lstm = pd.read_csv('Predictive models\\Data\\KerasLSTM_prediction.csv', header=0, index_col=0)
    stack_predict = np.vstack([arima['predict'].values, keras_lstm['prediction'].values]).T

    #FC NEXT
    #IF NOT PUT THE N NEXT STEPS AS MONTECARLO FC
    
    minmax = MinMaxScaler().fit(stack_predict.reshape(-1,1))
    stack_predict_scale = MinMaxScaler().fit_transform(stack_predict)
    real_price_scale = MinMaxScaler().fit_transform(np.array(real_price).reshape(-1,1))
    real_price_scale = np.where(np.isnan(real_price_scale), 0, real_price_scale)

    #Prepare for logistic regression
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
    plt.figure(figsize=(15,8))
    plt.plot(arima['predict'].values, color = 'red', label = 'arima')
    plt.plot(keras_lstm['prediction'].values, color = 'yellow', label = 'keras lstm')
    plt.plot(real_price, color = 'black', label = 'real price')
    plt.plot(ypred, color = 'green', label = 'stacked fc')
    plt.title('Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    #plt.savefig('Predictive models\\Prediction graphs\\XGB_Train_test_prediction.png')
    plt.show()

if __name__ == '__main__':
    main()

