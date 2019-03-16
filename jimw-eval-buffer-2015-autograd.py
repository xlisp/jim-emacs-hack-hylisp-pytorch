import torch as t
from torch.autograd import Variable

x = Variable(t.ones(2,2), requires_grad=True)
# tensor([[1., 1.],
#         [1., 1.]], requires_grad=True)

y = x.sum() #=> tensor(4., grad_fn=<SumBackward0>)

y.grad_fn #=> <SumBackward0 object at 0x102a53278>

y.backward()

x.grad
# tensor([[1., 1.],
#         [1., 1.]])

### grad在反向传播过程中累积的, 序列(句子)太长导致梯度爆炸=>LSTM
y.backward()
x.grad
# tensor([[2., 2.],
#         [2., 2.]])

y.backward()
x.grad
# tensor([[3., 3.],
#         [3., 3.]])

x.grad.data.zero_()
# tensor([[0., 0.],
#         [0., 0.]])

y.backward()
x.grad
# tensor([[1., 1.],
#         [1., 1.]])

x = Variable(t.ones(4,5)) # 4*5
y = t.cos(x)
# tensor([[0.5403, 0.5403, 0.5403, 0.5403, 0.5403],
#         [0.5403, 0.5403, 0.5403, 0.5403, 0.5403],
#         [0.5403, 0.5403, 0.5403, 0.5403, 0.5403],
#         [0.5403, 0.5403, 0.5403, 0.5403, 0.5403]])

t.cos(x.data)
# tensor([[0.5403, 0.5403, 0.5403, 0.5403, 0.5403],
#         [0.5403, 0.5403, 0.5403, 0.5403, 0.5403],
#         [0.5403, 0.5403, 0.5403, 0.5403, 0.5403],
#         [0.5403, 0.5403, 0.5403, 0.5403, 0.5403]])

