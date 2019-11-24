import torchvision.models.squeezenet as squeezenet
import torch.nn as nn


def get_squeezenet1_0():
    model = squeezenet.squeezenet1_0()
<<<<<<< HEAD
    model.classifier[1] = nn.Conv2d(512, 100, kernel_size=(1, 1), stride=(1, 1))
=======
    model.classifier[1] = nn.Conv2d(512, 70, kernel_size=(1, 1), stride=(1, 1))
>>>>>>> 2eefc0787a2de19b8aefd6ba423b430da88f5f81

    return model


def get_squeezenet1_1():
    model = squeezenet.squeezenet1_1()
<<<<<<< HEAD
    model.classifier[1] = nn.Conv2d(512, 100, kernel_size=(1, 1), stride=(1, 1))
=======
    model.classifier[1] = nn.Conv2d(512, 70, kernel_size=(1, 1), stride=(1, 1))
>>>>>>> 2eefc0787a2de19b8aefd6ba423b430da88f5f81
    return model
