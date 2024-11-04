

from Regression.ClassicRgs import Pls, Anngression, Svregression, ELM, Rfregression
from Regression.CNN import CNNTrain

def  QuantitativeAnalysis(model, X_train, X_test, y_train, y_test):

    if model == "PLSR":
        Rmse, R2, Mae = Pls(X_train, X_test, y_train, y_test)
    elif model == "ANN":
        Rmse, R2, Mae = Anngression(X_train, X_test, y_train, y_test)
    elif model == "SVR":
        Rmse, R2, Mae = Svregression(X_train, X_test, y_train, y_test)
    elif model == "ELM":
        Rmse, R2, Mae = ELM(X_train, X_test, y_train, y_test)
    elif model == "RF":
        Rmse, R2, Mae = Rfregression(X_train, X_test, y_train, y_test)
    elif model == "CNN":
        Rmse, R2, Mae = CNNTrain("LPCNet",X_train, X_test, y_train, y_test,  200)
    else:
        print("no this model of QuantitativeAnalysis")

    return Rmse, R2, Mae 