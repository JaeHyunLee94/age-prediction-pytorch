import torchvision.models.squeezenet as squeezenet
import torch.nn as nn


def get_squeezenet1_0():
    model = squeezenet.squeezenet1_0()
    model.fc = nn.Linear(100, 70)


    return model


def get_squeezenet1_1():
    model = squeezenet.squeezenet1_1()
    model.fc = nn.Linear(100, 70)
    return model
