import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from math import sqrt
from sklearn.metrics import mean_squared_error


def main():

    series = pd.read_csv('PredictiveModels\\Data\\MSFT.csv', header=0, index_col=0)

    #Scale data and split train and test
    minmax = MinMaxScaler().fit(series.iloc[:,4].values.reshape((-1,1)))
    df = MinMaxScaler().fit_transform(series.iloc[:, 0:5].values)
    df_scale = pd.DataFrame(df)

    n = 20    
    split_point = len(df_scale) - n
    dataset, validation = df_scale[0:split_point], df_scale[split_point:]

    dX_train = dataset.iloc[:, 0:5].values
    dy_train = dataset.iloc[:, 4:5].values
    dX_test = validation.iloc[:, 0:5].values
    dy_test = validation.iloc[:, 4:5].values

    '''
    
    #Feautures reduction (PCA)
    pca = PCA(0.95)
    dX_train = pca.fit_transform(dX_train)
    dX_test = pca.fit_transform(dX_test)

    '''

    #xgb (Gradient boost trees) parameters
    params_xgd = {
        'max_depth': 7,
        'objective': 'reg:logistic',
        'learning_rate': 0.05,
        'n_estimators': 10000
    }

    bst = xgb.XGBRegressor(**params_xgd)
    bst.fit(dX_train,dy_train, eval_set=[(dX_train,dy_train)], eval_metric='rmse', early_stopping_rounds=20, verbose=False)

    #Gradient boost prediction
    ypred = bst.predict(dX_test, ntree_limit=bst.best_ntree_limit)

    #Reverse tranformation
    def reverse_close(array):
        return minmax.inverse_transform(array.reshape((-1,1)))

    #Plot prediction vs test data
    plt.figure(figsize=(15,8))
    plt.plot(reverse_close(dy_test), color = 'red', label = 'Real Data')
    plt.plot(reverse_close(ypred), color = 'green', label = 'Predicted Data')
    plt.title('Gradient Boost Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('PredictiveModels\\Prediction graphs\\XGB_Train_test_prediction.png')
    plt.show()

    #Model error
    rmse = sqrt(mean_squared_error(reverse_close(dy_test), reverse_close(ypred)))
    print('Mean square error between train model and test data is: %.2f'%(rmse))



if __name__ == '__main__':
    main()