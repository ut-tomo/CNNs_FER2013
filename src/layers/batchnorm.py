import numpy as np

class BatchNorm:
    def __init__(self, num_channels, momentum=0.9, eps=1e-5):
        self.num_channels = num_channels
        self.momentum = momentum
        self.eps = eps

        self.gamma = np.ones((1, num_channels, 1, 1))
        self.beta = np.zeros((1, num_channels, 1, 1))

        self.running_mean = np.zeros((1, num_channels, 1, 1))
        self.running_var = np.ones((1, num_channels, 1, 1))

        # 中間データ（学習時）
        self.x_centered = None
        self.std_inv = None
        self.x_norm = None

        # 勾配
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, is_training=True):
        if is_training:
            mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            var = np.var(x, axis=(0, 2, 3), keepdims=True)
            std = np.sqrt(var + self.eps)
            x_centered = x - mean
            x_norm = x_centered / std

            self.x_centered = x_centered
            self.std_inv = 1. / std
            self.x_norm = x_norm

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            x_centered = x - self.running_mean
            x_norm = x_centered / np.sqrt(self.running_var + self.eps)
            self.x_norm = x_norm

        out = self.gamma * x_norm + self.beta
        return out

    def backward(self, dout):
        N, C, H, W = dout.shape
        x_norm = self.x_norm
        x_centered = self.x_centered
        std_inv = self.std_inv

        # 勾配
        self.dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
        self.dgamma = np.sum(dout * x_norm, axis=(0, 2, 3), keepdims=True)

        dx_norm = dout * self.gamma

        dx_norm = dx_norm.reshape(N, C, -1)
        x_centered = x_centered.reshape(N, C, -1)
        std_inv = std_inv.reshape(1, C, 1)

        M = dx_norm.shape[2]  # H * W
        dvar = np.sum(dx_norm * x_centered * -0.5 * (std_inv**3), axis=2, keepdims=True)
        dmean = np.sum(-dx_norm * std_inv, axis=2, keepdims=True) + dvar * np.mean(-2. * x_centered, axis=2, keepdims=True)
        dx = dx_norm * std_inv + dvar * 2 * x_centered / M + dmean / M

        dx = dx.reshape(N, C, H, W)
        return dx
