import numpy as np
import sys, os
sys.path.append(os.pardir)
from layers.conv2d import Conv2D
from layers.MaxPool import MaxPool
from layers.fullyconnected import FullyConnected
from layers.activations import Relu
from layers.batchnorm import BatchNorm
from utils.utils import flatten, unflatten


class ResidualBlock:
    def __init__(self, W1, b1, W2, b2):
        self.conv1 = Conv2D(W1, b1, stride=1, pad=1)
        self.bn1 = BatchNorm(W1.shape[0])
        self.act1 = Relu()
        self.conv2 = Conv2D(W2, b2, stride=1, pad=1)
        self.bn2 = BatchNorm(W2.shape[0])
        self.act2 = Relu()

    def forward(self, x):
        self.input = x
        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = self.act1.forward(out)
        out = self.conv2.forward(out)
        out = self.bn2.forward(out)
        out += self.input  # residual connection
        out = self.act2.forward(out)
        return out

    def backward(self, dout):
        dout = self.act2.backward(dout)
        dresidual = dout
        dout = self.bn2.backward(dout)
        dout = self.conv2.backward(dout)
        dout = self.act1.backward(dout)
        dout = self.bn1.backward(dout)
        dout = self.conv1.backward(dout)
        return dout + dresidual


class ResNet18:
    def __init__(self, params):
        self.params = params
        self.block1 = ResidualBlock(params['W1'], params['b1'], params['W2'], params['b2'])
        self.pool1 = MaxPool(pool_h=2, pool_w=2, stride=2, pad=0)

        self.block2 = ResidualBlock(params['W3'], params['b3'], params['W4'], params['b4'])
        self.pool2 = MaxPool(pool_h=2, pool_w=2, stride=2, pad=0)

        self.fc1 = FullyConnected(params['W5'], params['b5'])
        self.act = Relu()
        self.fc2 = FullyConnected(params['W6'], params['b6'])
        self.shape_before_flatten = None

    def forward(self, x):
        x = self.block1.forward(x)
        x = self.pool1.forward(x)
        x = self.block2.forward(x)
        x = self.pool2.forward(x)
        self.shape_before_flatten = x.shape
        x = flatten(x)
        x = self.fc1.forward(x)
        x = self.act.forward(x)
        x = self.fc2.forward(x)
        return x

    def backward(self, dout):
        grads = {}
        dout = self.fc2.backward(dout)
        grads['W6'] = self.fc2.dW
        grads['b6'] = self.fc2.db

        dout = self.act.backward(dout)
        dout = self.fc1.backward(dout)
        grads['W5'] = self.fc1.dW
        grads['b5'] = self.fc1.db

        dout = unflatten(dout, self.shape_before_flatten)

        dout = self.pool2.backward(dout)
        dout = self.block2.backward(dout)
        grads['W4'] = self.block2.conv2.dW
        grads['b4'] = self.block2.conv2.db
        grads['W3'] = self.block2.conv1.dW
        grads['b3'] = self.block2.conv1.db

        dout = self.pool1.backward(dout)
        dout = self.block1.backward(dout)
        grads['W2'] = self.block1.conv2.dW
        grads['b2'] = self.block1.conv2.db
        grads['W1'] = self.block1.conv1.dW
        grads['b1'] = self.block1.conv1.db

        return grads
