import libsvm
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    # 获取乳腺癌数据集
    raw_data = load_breast_cancer()
    
    # 打印数据信息
    print(f"数据维度: {raw_data.data.shape}")
    print(f"维度信息: {raw_data.feature_names}")
    print(f"预测信息: {raw_data.target_names}")
    print(f"预测维度信息: {raw_data.target.shape}")

    # 分割数据集(测试集30%)
    xtrain, xtest, ytrain, ytest = train_test_split(raw_data.data, raw_data.target, test_size=0.3, random_state=1)

    # 拟合模型 (C的含义是正则化强度的倒数)
    model = libsvm.SVC(C=10, kernel='linear')
    model.fit(xtrain, ytrain)

    # 计算模型精度
    ypredict = model.predict(xtest)
    acc = accuracy_score(ypredict, ytest)
    print(f"acc: {acc}")


if __name__ == '__main__':
    main()
