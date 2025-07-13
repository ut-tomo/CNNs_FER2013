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
#model_name = "vgg" 
model_name = "resnet"


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
    epochs=5,
    mini_batch_size=64,
    optimizer='adam',
    optimizer_param={'lr': 0.001},
    evaluate_sample_num_per_epoch=500,
    verbose=True
)
trainer.train()


np.savez(weight_file, **model.params)
print(f"weights saved to '{weight_file}'.")


plt.plot(trainer.train_acc_list, label="train acc")
plt.plot(trainer.test_acc_list,  label="test acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.title(f"Accuracy over epochs ({model_name})")
plt.savefig("accuracy_plot.png")
plt.show()
print("accuracy plot saved to 'accuracy_plot.png'.")
