from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
# 加载iris数据集
iris = load_iris()
X, y = iris.data, iris.target
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# 构建贝叶斯分类器
model = GaussianNB()
model.fit(X_train, y_train)
# 预测测试集
y_pred = model.predict(X_test)
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("贝叶斯分类器的准确率：", accuracy)

from sklearn.tree import DecisionTreeClassifier
# 构建决策树分类器
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)
# 预测测试集
y_pred_tree = tree_model.predict(X_test)
# 计算准确率
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print("决策树的准确率：", accuracy_tree)

from sklearn.linear_model import LogisticRegression
# 构建逻辑回归模型
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
# 预测测试集
y_pred_lr = lr_model.predict(X_test)
# 计算准确率
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("逻辑回归模型的准确率：", accuracy_lr)
