from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_wine, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from matplotlib import pyplot as plt


def classifier():
    # 原始数据
    raw_data = load_wine()

    # 打印数据信息
    print(f"数据维度: {raw_data.data.shape}")
    print(f"维度信息: {raw_data.feature_names}")
    print(f"预测信息: {raw_data.target_names}")
    print(f"预测维度信息: {raw_data.target.shape}")

    # 划分数据
    xtrain, xtest, ytrain, ytest = train_test_split(raw_data.data, raw_data.target, test_size=0.3, random_state=1)

    # 拟合
    model = XGBClassifier(n_estimators=20,max_depth=4,learning_rate=0.1, subsample=0.7, colsample_bytree=0.7)
    model.fit(xtrain, ytrain)

    # 打印每个特征的重要度
    for name, w in zip(raw_data.feature_names, model.feature_importances_):
        print(f'{name}: {w}')

    # 预测
    ypredict = model.predict(xtest)
    acc = accuracy_score(ytest, ypredict)
    print(f"acc: {acc}")


def regressor():
    ''' regressor '''
    raw_data = fetch_california_housing()

    # 打印数据信息
    print(f"数据维度: {raw_data.data.shape}")
    print(f"维度信息: {raw_data.feature_names}")
    print(f"预测信息: {raw_data.target_names}")
    print(f"预测维度信息: {raw_data.target.shape}")

    # 分割数据集 (30%的测试集)
    xtrain, xtest, ytrain, ytest = train_test_split(raw_data.data, raw_data.target, test_size=0.3, random_state=420)

    # 拟合
    model = XGBRegressor(C=1, kernel='rbf')
    model.fit(xtrain, ytrain)

    # 预测，以评估性能
    ypredict = model.predict(xtest)
    mse = mean_squared_error(ypredict, ytest)
    print(f'mse: {mse}')

    # 绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    index = range(len(ytest))
    index = sorted(index, key=lambda x: ytest[x])
    plt.plot(ypredict[index])
    plt.plot(ytest[index])
    plt.legend(['预测值', '真实值'])
    plt.show()

if __name__ == '__main__':
    classifier()
    regressor()
