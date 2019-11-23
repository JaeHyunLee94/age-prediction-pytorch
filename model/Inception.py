import torchvision.models.inception as inception
import torch.nn as nn


def get_inception_v3():
    model = inception.inception_v3(transform_input=True)
    model.fc = nn.Linear(2048, 70)
    return model

