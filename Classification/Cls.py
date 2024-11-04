
from Classification.ClassicCls import ANN, SVM, PLS_DA, RF
from Classification.CNN import CNN
from Classification.SAE import SAE

def  QualitativeAnalysis(model, X_train, X_test, y_train, y_test):

    if model == "PLS_DA":
        acc = PLS_DA(X_train, X_test, y_train, y_test)
    elif model == "ANN":
        acc = ANN(X_train, X_test, y_train, y_test)
    elif model == "SVM":
        acc = SVM(X_train, X_test, y_train, y_test)
    elif model == "RF":
        acc = RF(X_train, X_test, y_train, y_test)
    elif model == "CNN":
        acc = CNN(X_train, X_test, y_train, y_test, 16, 160, 4)
    elif model == "SAE":
        acc = SAE(X_train, X_test, y_train, y_test)
    else:
        print("no this model of QuantitativeAnalysis")

    return acc