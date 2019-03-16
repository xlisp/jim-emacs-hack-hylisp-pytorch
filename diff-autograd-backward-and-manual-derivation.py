import torch as t
from torch.autograd import Variable as V

# 原函数 y = x^2 * exp(x) => 导数: 2x*exp(x) + x^2*exp(x)

def f(x):
    y = x**2 * t.exp(x)
    return y

def gradf(x):
    dx = 2*x*t.exp(x) + x**2*t.exp(x)
    return dx

x = V(t.randn(3,4), requires_grad=True)
# tensor([[ 0.3410, -1.2885,  1.5217, -0.8072],
#         [-0.2007,  0.3642, -0.6599,  1.5859],
#         [-1.2894, -0.0107,  2.5028, -0.4159]], requires_grad=True)

y = f(x) # 原函数
# tensor([[0.4475, 0.1923, 0.5435, 0.0085],
#         [4.2587, 0.3752, 0.1253, 0.4714],
#         [0.2490, 0.4134, 0.0113, 0.0320]], grad_fn=<MulBackward0>)

print(gradf(x))
# tensor([[ 3.6872, 19.6365, 12.0056, -0.2913],
#         [ 4.7344, 70.6367,  3.6312, -0.1443],
#         [ 5.2468, -0.4181, -0.1732,  0.8044]], grad_fn=<AddBackward0>)

y.backward(t.ones(y.size()))

x.grad
# tensor([[ 3.6872, 19.6365, 12.0056, -0.2913],
#         [ 4.7344, 70.6367,  3.6312, -0.1443],
#         [ 5.2468, -0.4181, -0.1732,  0.8044]])

# 结论: gradf(x)手动求导 和 y.backward自动求导的结果是一样的

