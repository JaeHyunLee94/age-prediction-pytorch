import torchvision.models.vgg as vgg
import torch.nn as nn


def get_vgg11():
    model = vgg.vgg11(pretrained=False)

    model.classifier[6] = nn.Linear(4096,70)
    return model
