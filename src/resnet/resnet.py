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
    def __init__(self, W1, b1, W2, b2, stride=1, use_projection=False, W_proj=None, b_proj=None):
        self.conv1 = Conv2D(W1, b1, stride=stride, pad=1)
        self.bn1 = BatchNorm(W1.shape[0])
        self.act1 = Relu()

        self.conv2 = Conv2D(W2, b2, stride=1, pad=1)
        self.bn2 = BatchNorm(W2.shape[0])
        self.act2 = Relu()

        self.use_projection = use_projection
        if use_projection:
            self.proj_conv = Conv2D(W_proj, b_proj, stride=stride, pad=0)

    def forward(self, x):
        self.x = x
        out = self.act1.forward(self.bn1.forward(self.conv1.forward(x)))
        out = self.bn2.forward(self.conv2.forward(out))
        shortcut = self.proj_conv.forward(x) if self.use_projection else x
        out += shortcut
        return self.act2.forward(out)

    def backward(self, dout):
        """
        自信なし
        """
        dout = self.act2.backward(dout)
        dshortcut = dout

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

        # Stem
        self.conv1 = Conv2D(params['W0'], params['b0'], stride=2, pad=3)  # 7x7 conv
        self.bn1 = BatchNorm(params['W0'].shape[0])
        self.act1 = Relu()
        self.pool1 = AvePool(pool_h=3, pool_w=3, stride=2, pad=1)

        # Layer1
        self.block1_1 = BasicBlock(params['W1'], params['b1'], params['W2'], params['b2'])
        self.block1_2 = BasicBlock(params['W3'], params['b3'], params['W4'], params['b4'])

        # Layer2
        self.block2_1 = BasicBlock(
            params['W5'], params['b5'], params['W6'], params['b6'],
            stride=2, use_projection=True,
            W_proj=params['W_proj1'], b_proj=params['b_proj1']
        )
        self.block2_2 = BasicBlock(params['W7'], params['b7'], params['W8'], params['b8'])

        # Layer3
        self.block3_1 = BasicBlock(
            params['W9'], params['b9'], params['W10'], params['b10'],
            stride=2, use_projection=True,
            W_proj=params['W_proj2'], b_proj=params['b_proj2']
        )
        self.block3_2 = BasicBlock(params['W11'], params['b11'], params['W12'], params['b12'])

        # Layer4
        self.block4_1 = BasicBlock(
            params['W13'], params['b13'], params['W14'], params['b14'],
            stride=2, use_projection=True,
            W_proj=params['W_proj3'], b_proj=params['b_proj3']
        )
        self.block4_2 = BasicBlock(params['W15'], params['b15'], params['W16'], params['b16'])

        self.pool = None  # GAP
        self.fc = FullyConnected(params['W17'], params['b17'])

    def forward(self, x):
        x = self.act1.forward(self.bn1.forward(self.conv1.forward(x)))
        x = self.pool1.forward(x)

        x = self.block1_1.forward(x)
        x = self.block1_2.forward(x)

        x = self.block2_1.forward(x)
        x = self.block2_2.forward(x)

        x = self.block3_1.forward(x)
        x = self.block3_2.forward(x)

        x = self.block4_1.forward(x)
        x = self.block4_2.forward(x)

        h, w = x.shape[2], x.shape[3]
        self.pool = AvePool(pool_h=h, pool_w=w, stride=1, pad=0)
        x = self.pool.forward(x)

        self.shape_before_flatten = x.shape
        x = flatten(x)
        x = self.fc.forward(x)
        return x

    def backward(self, dout):
        grads = {}

        dout = self.fc.backward(dout)
        grads['W17'] = self.fc.dW
        grads['b17'] = self.fc.db

        dout = unflatten(dout, self.shape_before_flatten)
        dout = self.pool.backward(dout)

        for name, block in reversed([
            ('block4_2', self.block4_2),
            ('block4_1', self.block4_1),
            ('block3_2', self.block3_2),
            ('block3_1', self.block3_1),
            ('block2_2', self.block2_2),
            ('block2_1', self.block2_1),
            ('block1_2', self.block1_2),
            ('block1_1', self.block1_1),
        ]):
            dout = block.backward(dout)
            w_prefix = self._get_weight_prefix(name)
            grads[f'W{w_prefix + 1}'] = block.conv2.dW
            grads[f'b{w_prefix + 1}'] = block.conv2.db
            grads[f'W{w_prefix}'] = block.conv1.dW
            grads[f'b{w_prefix}'] = block.conv1.db
            if block.use_projection:
                proj_id = (w_prefix - 1) // 2
                grads[f'W_proj{proj_id}'] = block.proj_conv.dW
                grads[f'b_proj{proj_id}'] = block.proj_conv.db

        return grads

    def _get_weight_prefix(self, block_name):
        block_to_index = {
            'block1_1': 1, 'block1_2': 3,
            'block2_1': 5, 'block2_2': 7,
            'block3_1': 9, 'block3_2': 11,
            'block4_1': 13, 'block4_2': 15
        }
        return block_to_index[block_name]
