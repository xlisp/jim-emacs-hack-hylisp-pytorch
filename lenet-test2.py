import torch.nn as nn
import torch.nn.functional as F
import torch as t
from torch.autograd import Variable

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

## -------- test2 -----------

net.parameters() # => <generator object Module.parameters at 0x103fc78e0>

list(net.parameters()) ### 为什么这里的参数的梯度都不是零呢？
# [Parameter containing:
# tensor([[[[ 1.9928e-01,  8.3960e-02, -1.7374e-01, -8.5591e-02,  7.8552e-02],
#           [-9.2745e-02, -5.1972e-02, -3.6323e-02, -1.7810e-01,  1.9459e-01],
#           [-5.6182e-02,  1.0560e-01, -1.3356e-02, -2.8953e-02, -1.1588e-01],
#           [-1.6660e-01,  1.0511e-01,  6.0372e-02,  7.3975e-02, -5.4012e-02],
#           [ 1.3235e-01,  1.3098e-02,  1.9503e-01, -6.8204e-02,  1.3393e-01]]],
# 
# 
#         [[[ 1.1858e-01,  5.2488e-02, -2.7742e-02,  9.2655e-02,  1.4274e-01],
#           [ 8.5948e-02,  1.9312e-01, -1.6614e-01, -6.7132e-02, -1.9844e-01],
#           [-1.6751e-01, -1.9003e-01, -4.3400e-02,  1.7506e-01,  8.4618e-03]
# ....
#

len(list(net.parameters())) #=> 10

for name, parameters in net.named_parameters():
    name
    #print(name, ":", parameters.size())
# conv1.weight : torch.Size([6, 1, 5, 5])
# conv1.bias : torch.Size([6])
# conv2.weight : torch.Size([16, 6, 5, 5])
# conv2.bias : torch.Size([16])
# fc1.weight : torch.Size([120, 400])
# fc1.bias : torch.Size([120])
# fc2.weight : torch.Size([84, 120])
# fc2.bias : torch.Size([84])
# fc3.weight : torch.Size([10, 84])
# fc3.bias : torch.Size([10])
# fc3.bias : torch.Size([10])
# 

input = Variable(t.randn(1, 1, 32, 32))
# tensor([[[[ 1.5993, -0.1435,  0.5873,  ...,  0.8446, -0.4160,  0.5077],
#           [ 1.3105,  1.8664,  1.0908,  ..., -1.2539,  0.4122, -0.3313],
#           [-0.7693, -1.4102, -0.8371,  ..., -1.0799,  2.0081,  0.1142],
#           ...,
#           [ 0.4170, -0.9890, -0.1524,  ..., -0.7634,  0.3392,  0.5080],
#           [-0.1260,  0.9213,  0.7502,  ...,  1.1998, -0.5529, -0.7528],
#           [ 3.2332, -0.4970, -1.7693,  ..., -0.3741,  0.0906, -0.6051]]]])

out = net(input)
# tensor([[-0.1228, -0.0429,  0.0344, -0.0363, -0.0928, -0.0122, -0.0476,  0.1075,
#           0.0443,  0.0615]], grad_fn=<AddmmBackward>)

out.size() # torch.Size([1, 10])

list(net.parameters()) # 参数的梯度乱的数字

net.zero_grad() # 所有的参数的梯度清零

list(net.parameters())

out.backward(Variable(t.ones(1, 10))) # 反向传播

list(net.parameters())
# [Parameter containing:
# tensor([[[[ 0.1207,  0.1288,  0.1743, -0.1406,  0.1162],
#           [ 0.1404,  0.1841, -0.1656, -0.0141,  0.1341],
#           [ 0.0819,  0.0658, -0.0875,  0.0503,  0.1934],
#           [-0.1011,  0.0261, -0.0266, -0.1443, -0.0488],
#           [ 0.1807, -0.1206,  0.0559,  0.1602, -0.1682]]],
# ....

output = net(input)
net.conv1.bias.grad #=> 灌了数据之后的梯度 tensor([-0.0482, -0.0005,  0.0987, -0.0132,  0.0477,  0.0413])

target = Variable(t.arange(0, 10)) # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

criterion = nn.MSELoss()
# loss = criterion(output, target) ### book: 28.6064
# Traceback (most recent call last):
#   File "./jim-emacs-hack-hylisp-pytorch/lenet-test2.py", line 112, in <module>
#     loss = criterion(output, target)
#   File "/Users/clojure/.pyenv/versions/3.6.5/lib/python3.6/site-packages/torch/nn/modules/module.py", line 489, in __call__
#     result = self.forward(*input, **kwargs)
#   File "/Users/clojure/.pyenv/versions/3.6.5/lib/python3.6/site-packages/torch/nn/modules/loss.py", line 435, in forward
#     return F.mse_loss(input, target, reduction=self.reduction)
#   File "/Users/clojure/.pyenv/versions/3.6.5/lib/python3.6/site-packages/torch/nn/functional.py", line 2156, in mse_loss
#     ret = torch._C._nn.mse_loss(expanded_input, expanded_target, _Reduction.get_enum(reduction))
# RuntimeError: Expected object of scalar type Float but got scalar type Long for argument #2 'target'

net.zero_grad()
net.conv1.bias.grad # tensor([0., 0., 0., 0., 0., 0.]) ### 第一个卷积层的梯度全为零, 反向传播之前的梯度
list(net.parameters())
# [Parameter containing:
# tensor([[[[-0.1357, -0.0275, -0.0302,  0.0419,  0.0948],
#           [ 0.1346,  0.0387,  0.0103,  0.1737, -0.0979],
#           [ 0.1622, -0.1124,  0.1925,  0.1827,  0.0740],
#           [ 0.1621,  0.1814, -0.1963,  0.0894,  0.1650],
#           [ 0.0576, -0.0448, -0.1821,  0.0410,  0.1481]]],
