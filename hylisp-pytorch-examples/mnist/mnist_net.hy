(import 
 argparse
 torch
 [torch.nn :as nn]
 [torch.nn.functional :as F]
 [torch.optim :as optim]
 [torchvision [datasets transforms]])

(defclass Net [nn.Module]
  (defn __init__ [self] ;; __init__ & forward前向传播是继承nn.Module类必须定义的两个方法
    ((. (super Net self) --init--))
    (setv ;; 都是有训练参数的放在__init__方法里面
     self.conv1 (nn.Conv2d 1 20 5 1)
     self.conv2 (nn.Conv2d 20 50 5 1)
     self.fc1 (nn.Linear (* 4 4 50) 500)
     self.fc2 (nn.Linear 500 10)))
  ;; 前向传播"多层的计算图"做卷积操作并激活,全连接和flatten等; 反向传播计算图求其梯度
  (defn forward [self x]
    (-> x
        (self.conv1)
        (F.relu)
        ((fn [x]
           (F.max_pool2d x 2 2)))
        (self.conv2)
        (F.relu)
        ((fn [x]
           (F.max_pool2d x 2 2)))
        ((fn [x]
           (x.view -1 (* 4 4 50))))
        (self.fc1)
        (F.relu)
        (self.fc2)
        ((fn [x]
           (F.log_softmax x :dim 1)))))
  )

