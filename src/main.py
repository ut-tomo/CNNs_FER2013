import numpy as np
import matplotlib.pyplot as plt
from layers.loss import SoftmaxCrossEntropyLoss
from train.trainer import Trainer
from utils.utils import *


from vgg.vgg import VGGNet
from resnet.resnet import ResNet18
from lenet.lenet import LeNet
from utils.utils import init_lenet_params, init_vgg_params, init_resnet_params


#model_name = "lenet"
model_name = "vgg" 
#model_name = "resnet"


x_train, t_train, x_test, t_test = load_fer2013()


if model_name == "lenet":
    params = init_lenet_params()
    model = LeNet(params)
    weight_file = "trained_lenet_weights.npz"
elif model_name == "vgg":
    params = init_vgg_params()
    model = VGGNet(params)
    weight_file = "trained_vgg_weights.npz"
elif model_name == "resnet":
    params = init_resnet_params()
    model = ResNet18(params)
    weight_file = "trained_resnet_weights.npz"
else:
    raise ValueError("Unknown model name")



loss_fn = SoftmaxCrossEntropyLoss()


trainer = Trainer(
    network=model,
    loss_fn=loss_fn,
    x_train=x_train, t_train=t_train,
    x_test=x_test,   t_test=t_test,
    epochs=300,
    mini_batch_size=64,
    optimizer='adam',
    optimizer_param={'lr': 0.001},
    evaluate_sample_num_per_epoch=500,
    verbose=True
)
trainer.train()


np.savez(weight_file, **model.params)
print(f"weights saved to '{weight_file}'.")


def plot_accuracy(epoch_limit, train_acc, test_acc, model_name):
    plt.figure()
    plt.plot(train_acc[:epoch_limit], label="train acc")
    plt.plot(test_acc[:epoch_limit], label="test acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.title(f"Accuracy over epochs  {model_name}")
    filename = f"accuracy_plot_{model_name}_epoch{epoch_limit}.png"
    plt.savefig(filename)
    plt.show()
    print(f"accuracy plot saved to '{filename}'.")


# 各エポック数で可視化
plot_accuracy(100, trainer.train_acc_list, trainer.test_acc_list, model_name)
plot_accuracy(200, trainer.train_acc_list, trainer.test_acc_list, model_name)
plot_accuracy(300, trainer.train_acc_list, trainer.test_acc_list, model_name)
