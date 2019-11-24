import torch.nn as nn
import torchvision.models.resnet as resnet
import torch


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1):
        super(BasicBlock, self).__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),

            nn.Conv2d(out_channel, out_channel * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel * BasicBlock.expansion)
        )
        self.short_cut = nn.Sequential()

        if stride != 1 or in_channel != self.expansion * out_channel:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel * self.expansion)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.residual_function(x) + self.short_cut(x))


class BottleNeck(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1):
        super(BottleNeck, self).__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),

            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),

            nn.Conv2d(out_channel, out_channel * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel * self.expansion),

        )

        self.short_cut = nn.Sequential()
        self.relu = nn.ReLU()

        if stride != 1 or in_channel != out_channel * self.expansion:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel * self.expansion)
            )

    def forward(self, x):
        return self.relu(self.residual_function(x) + self.short_cut(x))


class ResNet(nn.Module):

    def __init__(self, block_type, layer_channel, stride=1):
        super(ResNet, self).__init__()
        self.in_channel = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        layer_arr = []

        for out_channel in layer_channel:
            layer_arr.append(block_type(self.in_channel, out_channel * block_type.expansion, stride=stride))
            self.in_channel = out_channel * block_type.expansion

        self.block_layer = nn.Sequential(*layer_arr)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.in_channel, 70)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block_layer(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.relu(x)
        return x


def get_resnet6():
    model = ResNet(BasicBlock, [64, 32, 32])
    return model


def get_resnet12():
    model = ResNet(BottleNeck, [64, 128, 128, 128, 64])
    return model


def get_resnet18():
    model = resnet.resnet18(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 100)
    return model


def get_resnet34():
    model = resnet.resnet34(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 100)
    return model

def get_restnet55():
    model=resnet.resnet50(pretrained=False)
    in_features=model.fc.in_features
    model.fc=nn.Linear(in_features,100)
    return model
