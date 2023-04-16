from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt


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

    # 数据归一化
    std = StandardScaler()
    xtrain = std.fit_transform(xtrain)

    # 拟合
    model = SVR(C=1, kernel='rbf')
    model.fit(xtrain, ytrain)

    # 预测，以评估性能
    xtest = std.transform(xtest)
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
    main()
