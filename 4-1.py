import torch
import torch.nn as nn
# 定义一个继承自nn.Module的类来构建神经网络
class FCNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(FCNet, self).__init__()
        # 定义全连接层
        self.fc = nn.Linear(input_size, output_size)
    def forward(self, x):
        # 在forward方法中定义前向传播过程
        x = self.fc(x)
        return x
# 创建一个FCNet实例
input_size = 10
output_size = 5
fc_net = FCNet(input_size, output_size)
# 打印网络结构
print(fc_net)
# 随机生成输入数据
input_data = torch.randn(1, input_size)
# 进行前向传播
output = fc_net(input_data)
print(output)
