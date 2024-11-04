import matplotlib.pyplot as plt



def plot_scatter1(y_test, y_pred):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, edgecolors='b', facecolors='none', label='Predicted vs Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Prediction')
    plt.legend()
    plt.show()

def plot_scatter2(y_test, y_pred, y_train, y_pred_train, model_name='Model'):
    plt.figure(figsize=(8, 6))
    # 建模集用圆圈表示
    plt.scatter(y_train, y_pred_train, edgecolors='b', facecolors='none', linewidths=2, s=50,
                label='Train Predicted')
    # 预测集用三角形表示，增加边框宽度为2，调整大小为50
    plt.scatter(y_test, y_pred, marker='^', edgecolors='r', facecolors='none', linewidths=2, s=50,
                label='Test Predicted ')
    # 将y=x的线变成黑色，并增加线宽为2
    plt.plot([min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())],
             [min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())], 'k--', lw=3)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{model_name} Prediction')
    plt.legend()
    plt.show()