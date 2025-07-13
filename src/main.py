import numpy as np
from lenet.lenet import LeNet
from layers.loss import SoftmaxCrossEntropyLoss
from train.trainer import Trainer
from utils.utils import *
import matplotlib.pyplot as plt

x_train, t_train, x_test, t_test = load_fer2013()


params = init_lenet_params()
model  = LeNet(params)                      
loss_fn = SoftmaxCrossEntropyLoss()

trainer = Trainer(
    network=model,
    loss_fn=loss_fn,
    x_train=x_train, t_train=t_train,
    x_test=x_test,   t_test=t_test,
    epochs=200,
    mini_batch_size=64,
    optimizer='adam',
    optimizer_param={'lr':0.001},
    evaluate_sample_num_per_epoch=500,
    verbose=True
)
trainer.train()

np.savez("trained_lenet_weights.npz", **model.params)
print("weights saved to 'trained_lenet_weights.npz'.")


#描画
plt.plot(trainer.train_acc_list, label="train acc")
plt.plot(trainer.test_acc_list,  label="test acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.title("Accuracy over epochs")
plt.savefig("accuracy_plot.png")
plt.show()
print("accuracy plot saved to 'accuracy_plot.png'.")