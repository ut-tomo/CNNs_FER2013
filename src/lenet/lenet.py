import numpy as np
import sys, os
sys.path.append(os.pardir)
from layers.conv2d import Conv2D
from layers.AvePool import AvePool
from layers.fullyconnected import FullyConnected
from layers.activations import Relu
from layers.batchnorm import BatchNorm
from utils.utils import flatten, unflatten

class LeNet:
    def __init__(self, params):
        """
        params: 辞書形式で各レイヤーの W, b を持つ
        """
        self.params = params
        self.conv1 = Conv2D(params['W1'], params['b1'], stride=1, pad=0)
        self.bn1 = BatchNorm(params['W1'].shape[0])
        self.act1 = Relu()
        self.pool1 = AvePool(pool_h=2, pool_w=2, stride=2, pad=0)

        self.conv2 = Conv2D(params['W2'], params['b2'], stride=1, pad=0)
        self.bn2 = BatchNorm(params['W2'].shape[0])
        self.act2 = Relu()
        self.pool2 = AvePool(pool_h=2, pool_w=2, stride=2, pad=0)

        self.fc1 = FullyConnected(params['W3'], params['b3'])  # (1296 → 84)
        self.act3 = Relu()
        self.fc2 = FullyConnected(params['W4'], params['b4'])  # (84 → 7)

        self.shape_before_flatten = None

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.bn1.forward(x)
        x = self.act1.forward(x)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        x = self.bn2.forward(x)
        x = self.act2.forward(x)
        x = self.pool2.forward(x)

        self.shape_before_flatten = x.shape 
        x = flatten(x)

        x = self.fc1.forward(x)
        x = self.act3.forward(x)
        x = self.fc2.forward(x)
        return x 

    def backward(self, dout):
        grads = {}

        dout = self.fc2.backward(dout)
        grads['W4'] = self.fc2.dW
        grads['b4'] = self.fc2.db

        dout = self.act3.backward(dout)

        dout = self.fc1.backward(dout)
        grads['W3'] = self.fc1.dW
        grads['b3'] = self.fc1.db

        dout = unflatten(dout, self.shape_before_flatten)

        # Conv2
        dout = self.pool2.backward(dout)
        dout = self.act2.backward(dout)
        dout = self.bn2.backward(dout)
        dout = self.conv2.backward(dout)
        grads['W2'] = self.conv2.dW
        grads['b2'] = self.conv2.db

        # Conv1
        dout = self.pool1.backward(dout)
        dout = self.act1.backward(dout)
        dout = self.bn1.backward(dout)
        dout = self.conv1.backward(dout)
        grads['W1'] = self.conv1.dW
        grads['b1'] = self.conv1.db

        return grads
