from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error

def main():
    ''' main '''
    raw_data = fetch_california_housing()

    # 打印数据信息
    print(f"数据维度: {raw_data.data.shape}")
    print(f"维度信息: {raw_data.feature_names}")
    print(f"预测信息: {raw_data.target_names}")
    print(f"预测维度信息: {raw_data.target.shape}")

    # 分割数据集 (30%的测试集)
    xtrain, xtest, ytrain, ytest = train_test_split(raw_data.data, raw_data.target, test_size=0.3, random_state=420)

    # 拟合
    model = LinearRegression()
    # model = Ridge()
    # model = Lasso()
    model.fit(xtrain, ytrain)

    # 打印模型参数
    print("------------------")
    for name, w in zip(raw_data.feature_names, model.coef_):
        print(f'{name}: {w:.4f}')
    print(f"b: {model.intercept_}")

    # 预测，以评估性能
    ypredict = model.predict(xtest)
    mse = mean_squared_error(ypredict, ytest)
    print(f'mse: {mse}')


if __name__ == '__main__':
    main()
