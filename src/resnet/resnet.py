import numpy as np
import sys, os
sys.path.append(os.pardir)
from layers.conv2d import Conv2D
from layers.AvePool import AvePool
from layers.fullyconnected import FullyConnected
from layers.batchnorm import BatchNorm
from layers.activations import Relu
from utils.utils import flatten, unflatten


class BasicBlock:
    def __init__(self, W1, b1, W2, b2, use_projection=False, W_proj=None, b_proj=None):
        self.conv1 = Conv2D(W1, b1, stride=1, pad=1)
        self.bn1 = BatchNorm(W1.shape[0])
        self.act1 = Relu()

        self.conv2 = Conv2D(W2, b2, stride=1, pad=1)
        self.bn2 = BatchNorm(W2.shape[0])

        self.act2 = Relu()
        self.use_projection = use_projection

        if use_projection:
            self.proj_conv = Conv2D(W_proj, b_proj, stride=1, pad=0)

    def forward(self, x):
        self.x = x
        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = self.act1.forward(out)

        out = self.conv2.forward(out)
        out = self.bn2.forward(out)

        shortcut = self.proj_conv.forward(x) if self.use_projection else x
        out += shortcut
        out = self.act2.forward(out)
        return out

    def backward(self, dout):
        dout = self.act2.backward(dout)
        dshortcut = dout  # for residual

        dout = self.bn2.backward(dout)
        dout = self.conv2.backward(dout)

        dout = self.act1.backward(dout)
        dout = self.bn1.backward(dout)
        dout = self.conv1.backward(dout)

        if self.use_projection:
            dshortcut = self.proj_conv.backward(dshortcut)

        return dout + dshortcut


class ResNet18:
    def __init__(self, params):
        self.params = params

        # Block 1: 64→64
        self.block1 = BasicBlock(params['W1'], params['b1'], params['W2'], params['b2'])

        # Block 2: 64→128 (use projection)
        self.block2 = BasicBlock(
            params['W3'], params['b3'], params['W4'], params['b4'],
            use_projection=True, W_proj=params['W_proj1'], b_proj=params['b_proj1']
        )

        # Block 3: 128→256 (use projection)
        self.block3 = BasicBlock(
            params['W5'], params['b5'], params['W6'], params['b6'],
            use_projection=True, W_proj=params['W_proj2'], b_proj=params['b_proj2']
        )

        # Block 4: 256→512 (use projection)
        self.block4 = BasicBlock(
            params['W7'], params['b7'], params['W8'], params['b8'],
            use_projection=True, W_proj=params['W_proj3'], b_proj=params['b_proj3']
        )

        self.pool = AvePool(pool_h=6, pool_w=6, stride=1, pad=0)  # Global AvgPool for 6x6

        self.fc = FullyConnected(params['W9'], params['b9'])  # 512 → 7

    def forward(self, x):
        x = self.block1.forward(x)
        x = self.block2.forward(x)
        x = self.block3.forward(x)
        x = self.block4.forward(x)
        self.shape_before_flatten = x.shape
        x = self.pool.forward(x)
        x = flatten(x)
        x = self.fc.forward(x)
        return x

    def backward(self, dout):
        grads = {}

        dout = self.fc.backward(dout)
        grads['W9'] = self.fc.dW
        grads['b9'] = self.fc.db

        dout = unflatten(dout, self.shape_before_flatten)
        dout = self.pool.backward(dout)

        dout = self.block4.backward(dout)
        grads['W8'] = self.block4.conv2.dW
        grads['b8'] = self.block4.conv2.db
        grads['W7'] = self.block4.conv1.dW
        grads['b7'] = self.block4.conv1.db
        grads['W_proj3'] = self.block4.proj_conv.dW
        grads['b_proj3'] = self.block4.proj_conv.db

        dout = self.block3.backward(dout)
        grads['W6'] = self.block3.conv2.dW
        grads['b6'] = self.block3.conv2.db
        grads['W5'] = self.block3.conv1.dW
        grads['b5'] = self.block3.conv1.db
        grads['W_proj2'] = self.block3.proj_conv.dW
        grads['b_proj2'] = self.block3.proj_conv.db

        dout = self.block2.backward(dout)
        grads['W4'] = self.block2.conv2.dW
        grads['b4'] = self.block2.conv2.db
        grads['W3'] = self.block2.conv1.dW
        grads['b3'] = self.block2.conv1.db
        grads['W_proj1'] = self.block2.proj_conv.dW
        grads['b_proj1'] = self.block2.proj_conv.db

        dout = self.block1.backward(dout)
        grads['W2'] = self.block1.conv2.dW
        grads['b2'] = self.block1.conv2.db
        grads['W1'] = self.block1.conv1.dW
        grads['b1'] = self.block1.conv1.db

        return grads
