import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
import hpelm
import joblib
import dill as pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from Evaluate.RgsEvaluate import ModelRgsevaluate
from Plot.fitted_curve import  plot_scatter1,plot_scatter2
def Pls( X_train, X_test, y_train, y_test):


    model = PLSRegression(n_components=3)
    # fit the model
    model.fit(X_train, y_train)

    # predict the values
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    Rmse, R2, Mae = ModelRgsevaluate(y_pred_test, y_test)

    plot_scatter2(y_test, y_pred_test.ravel(), y_train, y_pred_train.ravel(), model_name='PLS')
    # 保存模型
    # joblib.dump(model, 'ChlorophylB_pls_model.joblib')

    return Rmse, R2, Mae

def Rfregression(X_train, X_test, y_train, y_test):
    # 初始化随机森林回归模型
    model = RandomForestRegressor(n_estimators=600, random_state=123)
    model.fit(X_train, y_train)

    # 预测值
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # 打印训练数据的指标
    Rmse_train, R2_train, Mae_train = ModelRgsevaluate(y_pred_train, y_train)
    print(f"Training Set - RMSEC: {Rmse_train}, R2C: {R2_train}, MAEC: {Mae_train}")

    # 打印预测数据的指标
    Rmse, R2, Mae = ModelRgsevaluate(y_pred_test, y_test)

    # 绘制散点图
    plot_scatter2(y_test, y_pred_test.ravel(), y_train, y_pred_train.ravel(), model_name='RF')

    # # 保存模型到磁盘
    # joblib.dump(model, 'rf_TPC_model.joblib')

    return Rmse, R2, Mae

def Svregression(X_train, X_test, y_train, y_test):


    # model = SVR(C=5, gamma=1e-07, kernel='linear') # 线性
    model = SVR(C=5, gamma='scale', kernel='rbf')  # 非线性
    model.fit(X_train, y_train)

    # predict the values
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    Rmse, R2, Mae = ModelRgsevaluate(y_pred_test, y_test)

    plot_scatter2(y_test, y_pred_test.ravel(), y_train, y_pred_train.ravel(), model_name='SVR')
    # 保存模型
    # joblib.dump(model, 'SVR_ChlorophylB_pls_model.joblib')
    return Rmse, R2, Mae

def Anngression(X_train, X_test, y_train, y_test):

    model = MLPRegressor(
        hidden_layer_sizes=(20, 20), activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
        learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=400, shuffle=True,
        random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.fit(X_train, y_train)

    # predict the values
    y_pred = model.predict(X_test)
    Rmse, R2, Mae = ModelRgsevaluate(y_pred, y_test)

    return Rmse, R2, Mae

def ELM(X_train, X_test, y_train, y_test ):

    model = hpelm.ELM(X_train.shape[1], 1)
    model.add_neurons(20, 'sigm')  #非线性激活函数


    model.train(X_train, y_train, 'r')
    # predict the values
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)


    Rmse, R2, Mae = ModelRgsevaluate(y_pred_test, y_test)

    plot_scatter2(y_test, y_pred_test.ravel(), y_train, y_pred_train.ravel(),model_name = 'ELM')

    # Saving model
    # model.save("ELM_ChlorophylB_pls_model.h5")

    return Rmse, R2, Mae