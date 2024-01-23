import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

# 加载Iris数据集
iris = load_iris()
X, y = iris.data, iris.target

# 数据预处理
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)#np.mean求列平均值,np.std求列（axis=0）方差
X = X.reshape(-1, 1, 4)  # 调整输入形状以适应CNN,由[150,4]变为[150,1,4](150*4/(1*4))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将NumPy数组转换为PyTorch张量,pytorch要使用张量
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).long()
X_test = torch.from_numpy(X_test).float()

# 定义FC神经网络模型
class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(4, 3)#全连接

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 平铺输入与reshape类似，[120,1,4]变[120,4]
        x = self.fc1(x)
        return x

# 定义CNN神经网络模型/
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 3, kernel_size=1)#卷积,3个卷积，一个卷积长1，宽1.[120,1,4]变[120,3,4]
        self.avgpool = nn.AdaptiveAvgPool1d(1)#自适应平均池化，[120,3,4]变[120,3,1]

    def forward(self, x):
        x = self.conv1(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 平铺输入[120,3,1]变[120,3]
        return x

# 初始化FC和CNN模型
fc_model = FCNet()
cnn_model = CNNNet()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
fc_optimizer = optim.SGD(fc_model.parameters(), lr=0.1)#第一个为要调整的参数，第二个为学习率
cnn_optimizer = optim.SGD(cnn_model.parameters(), lr=0.1)

# 训练FC模型
fc_model.train()#训练模式
for epoch in range(100):
    fc_optimizer.zero_grad()#清除之前的梯度
    output = fc_model(X_train)
    loss = criterion(output, y_train)#计算损失函数
    loss.backward()#求梯度
    fc_optimizer.step()#根据梯度调整参数

# 训练CNN模型
cnn_model.train()
for epoch in range(100):
    cnn_optimizer.zero_grad()
    output = cnn_model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    cnn_optimizer.step()

# 在测试集上进行预测
fc_model.eval()#测试模式
cnn_model.eval()
fc_predictions = torch.argmax(fc_model(X_test), dim=1).numpy()#argmax通过dim指示的维度求最大值来降维
cnn_predictions = torch.argmax(cnn_model(X_test), dim=1).numpy()

# 构建SVM分类器
svm_classifier = SVC()
svm_classifier.fit(X_train.view(X_train.size(0), -1).numpy(), y_train.numpy())
svm_predictions = svm_classifier.predict(X_test.view(X_test.size(0), -1).numpy())

# 可视化结果
plt.scatter(X_test[:,0,2], X_test[:,0,3], c=y_test)#0表示第一个时间片，即初始时观测到的值，2表示第三个特征
plt.title("True labels")
plt.show()

plt.scatter(X_test[:, 0, 2], X_test[:, 0, 3], c=fc_predictions)
plt.title("FC Predictions")
plt.show()

plt.scatter(X_test[:, 0, 2], X_test[:, 0, 3], c=cnn_predictions)
plt.title("CNN Predictions")
plt.show()

plt.scatter(X_test[:, 0, 2], X_test[:, 0, 3], c=svm_predictions)
plt.title("SVM Predictions")
plt.show()









