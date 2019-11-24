import torchvision.models.vgg as vgg
import torch.nn as nn


def get_vgg11():
    model = vgg.vgg11(pretrained=False)

<<<<<<< HEAD
    model.classifier[6] = nn.Linear(4096,100)
=======
    model.classifier[6] = nn.Linear(4096,70)
>>>>>>> 2eefc0787a2de19b8aefd6ba423b430da88f5f81
    return model
