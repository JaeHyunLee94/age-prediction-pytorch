import torchvision.models.densenet as densenet
import torch.nn as nn


def get_densenet121():
    model = densenet.densenet121()
    model.classifier = nn.Linear(1024, 100)
    return model


def get_densenet161():
    model = densenet.densenet161()
    model.classifier = nn.Linear(1024, 100)
    return model
