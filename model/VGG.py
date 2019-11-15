import torchvision.models.vgg as vgg
import torch.nn as nn


def get_vgg11():
    model = vgg.vgg11(pretrained=False)

    model.fc = nn.Linear(512, 100)
    return model
