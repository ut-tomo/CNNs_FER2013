import numpy as np

class SoftmaxCrossEntropyLoss:
    def __init__(self):
        self.loss = None  # 損失値
        self.y = None  # softmax後の出力
        self.t = None  # 正解ラベル（整数）

    def softmax(self, x):
        """
        x: (N, C)
        return: (N, C)
        """
        x = x - np.max(x, axis=1, keepdims=True)  # overflow対策
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def cross_entropy_loss(self, y, t):
        """
        y: (N, C) softmax後の出力
        t: (N,) 整数ラベル
        return: スカラーの平均損失
        """
        N = y.shape[0]
        log_probs = -np.log(y[np.arange(N), t] + 1e-7)
        loss = np.mean(log_probs)
        return loss

    def forward(self, x, t):
        """
        x: (N, C) 生の出力
        t: (N,)  整数ラベル
        return: スカラーの平均損失
        """
        self.y = self.softmax(x)
        self.t = t
        self.loss = self.cross_entropy_loss(self.y, self.t)
    
        return self.loss
    
    def backward(self):
        batch_size = self.t.shape[0]
        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx /= batch_size
        return dx
