import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# 加载Iris数据集
iris = load_iris()
X = iris.data
y = iris.target
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# 对每个类别进行回归并计算准确率
accuracies = []
for class_label in np.unique(y):
    # 提取当前类别的样本和标签，==为true提取
    X_class = X_train[y_train == class_label]
    y_class = y_train[y_train == class_label]
    # 使用最小二乘法进行线性回归
    X_class = np.c_[np.ones(len(X_class)), X_class]  # 给每个样本加上1，相当于添加截距项或者说bias 1。
    coef = np.linalg.lstsq(X_class, y_class, rcond=None)[0]#使用最小二乘法来计算由输入特征矩阵 X_class 和目标向量 y_class 定义的线性方程的最小二乘解,即元组[0]。它返回一个包含多个元素的元组
    # 在测试集上进行预测
    X_test_class = X_test[y_test == class_label]
    y_pred = np.dot(np.c_[np.ones(len(X_test_class)), X_test_class], coef)
    y_pred = np.round(y_pred).astype(int)  # 四舍五入为整数
    # 计算当前类别的准确率
    accuracy = np.mean(y_pred == y_test[y_test == class_label])
    print(accuracy)
    accuracies.append(accuracy)
    # 可视化拟合结果
    plt.scatter(X_test_class[:, 0], y_test[y_test == class_label], color='blue', label='True')
    plt.scatter(X_test_class[:, 0], y_pred, color='red', label='Predicted')
    plt.xlabel('Feature')
    plt.ylabel('Class')
    plt.title(f'Linear Regression for Class {class_label}')
    plt.legend()
    plt.show()
# 计算总体准确率
total_accuracy = np.mean(accuracies)
print(f"Total Accuracy: {total_accuracy}")
