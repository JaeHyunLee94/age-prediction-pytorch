import torchvision.models.densenet as densenet
import torch.nn as nn


def get_densenet121():
    model = densenet.densenet121()
<<<<<<< HEAD
    model.classifier = nn.Linear(1024, 100)
=======
    model.classifier = nn.Linear(1024, 70)
>>>>>>> 2eefc0787a2de19b8aefd6ba423b430da88f5f81
    return model


def get_densenet161():
    model = densenet.densenet161()
<<<<<<< HEAD
    model.classifier = nn.Linear(1024, 100)
=======
    model.classifier = nn.Linear(1024, 70)
>>>>>>> 2eefc0787a2de19b8aefd6ba423b430da88f5f81
    return model
