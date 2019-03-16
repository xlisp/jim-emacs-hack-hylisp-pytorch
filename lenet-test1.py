import torch.nn as nn
import torch.nn.functional as F
# 你在手打一遍别人的代码的时候,就是重新开始用自己的方式编码: 不断的自我感知
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        ## 所有的需要重新"摆好"的参数都放在__init__函数里面
        self.conv1 = nn.Conv2d(1, 6, 5) # 1是单通道(黑白颜色), 6表示输出通道数, 5表示5*5的卷积核
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全连接层 , y = Wx + b
        self.fc1 = nn.Linear(16*5*5, 120) # 输入 + 输出 + 本层管道属性 16*5*5->120->84->10, 10是最后的分类数
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # 前向传播就是大数据的数据流管道+不可变数据结构思想: map-reduce-partition => lenet(cnn), rnn, lstm网络结构图等都是前向传播网络,接收输入,经过层层传递几何运算(线性代数的几何变换),得到输出
    def forward(self, x):
        # 卷积 -> 激活 -> 池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # reshape, -1表示自适应 (? 除了第一个? 网上的说法也半对半错)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
# Net(
#   (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
#   (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
#   (fc1): Linear(in_features=400, out_features=120, bias=True)
#   (fc2): Linear(in_features=120, out_features=84, bias=True)
#   (fc3): Linear(in_features=84, out_features=10, bias=True) ### 数据管道最后输出的特征out_features=10
# )
#

