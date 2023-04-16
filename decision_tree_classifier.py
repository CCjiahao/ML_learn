from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
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
    model = DecisionTreeClassifier()
    model.fit(xtrain, ytrain)

    # 打印每个特征的重要度
    for name, w in zip(raw_data.feature_names, model.feature_importances_):
        print(f'{name}: {w}')

    # 预测
    ypredict = model.predict(xtest)
    acc = accuracy_score(ytest, ypredict)
    print(f"acc: {acc}")


if __name__ == '__main__':
    main()
