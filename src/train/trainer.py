import numpy as np
import sys, os
sys.path.append(os.pardir)
from train.optimizer import *

class Trainer:
    def __init__(self, network, loss_fn, 
                 x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr':0.01},
                 evaluate_sample_num_per_epoch=None, verbose=True):
        self.network = network
        self.loss_fn = loss_fn
        self.verbose = verbose

        self.x_train, self.t_train = x_train, t_train
        self.x_test,  self.t_test  = x_test,  t_test

        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        opt_dict = {'sgd':SGD, 'momentum':Momentum, 'nesterov':Nesterov,
                    'adagrad':AdaGrad, 'rmsprop':RMSprop, 'adam':Adam}
        self.optimizer = opt_dict[optimizer.lower()](**optimizer_param)

        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size // mini_batch_size, 1)
        self.max_iter = self.epochs * self.iter_per_epoch
        self.current_iter = 0
        self.current_epoch = 0

        self.train_loss_list, self.train_acc_list, self.test_acc_list = [], [], []



    def _accuracy(self, x, t):
        y = self.network.forward(x)
        pred = np.argmax(y, axis=1)
        if t.ndim != 1: 
            t = np.argmax(t, axis=1)
        return np.mean(pred == t)


    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch, t_batch = self.x_train[batch_mask], self.t_train[batch_mask]

        logits = self.network.forward(x_batch)
        loss   = self.loss_fn.forward(logits, t_batch)
        self.train_loss_list.append(loss)


        dout  = self.loss_fn.backward() 
        grads = self.network.backward(dout)

        self.optimizer.update(self.network.params, grads)

        if self.verbose:
            print(f"[iter {self.current_iter+1}] loss: {loss:.4f}")

        if (self.current_iter + 1) % self.iter_per_epoch == 0:
            self.current_epoch += 1
            if self.evaluate_sample_num_per_epoch:
                t = self.evaluate_sample_num_per_epoch
                train_acc = self._accuracy(self.x_train[:t], self.t_train[:t])
                test_acc  = self._accuracy(self.x_test[:t],  self.t_test[:t])
            else:
                train_acc = self._accuracy(self.x_train, self.t_train)
                test_acc  = self._accuracy(self.x_test,  self.t_test)

            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose:
                print(f"=== epoch:{self.current_epoch}, "
                      f"train acc:{train_acc:.3f}, test acc:{test_acc:.3f} ===")

        self.current_iter += 1

    def train(self):
        for _ in range(self.max_iter):
            self.train_step()

        final_test_acc = self._accuracy(self.x_test, self.t_test)
        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print(f"test acc: {final_test_acc:.3f}")