import torch
import torch.nn as nn
import pickle


def resnet5(layers):
    model = ResNet(layers)
    # 학습 완료된 모델있으면 불러오기??
    # 아니면 학습 시키고 불러오기

    return model


class ResNet(nn.Module):

    def __init__(self, layers):
        super(ResNet, self).__init__()
        channel = 16
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.stack1 = self.make_stack()
        self.stack2 = self.make_stack()
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * Bottleneck.expansion, 100)
        self.is_trained = False

    def forward(self, x):
        out = self.conv1(x)
        out = self.stack1(out)
        out = self.stack2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class BottleNeck(nn.Module):

    def __init__(self, in_channel, mid_channel, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channel)
        self.conv2 = nn.Conv2d(mid_channel, mid_channel, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channel)
        self.conv3 = nn.Conv2d(mid_channel, in_channel, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(in_channel)
        self.reLu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.reLu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.reLu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.reLu(out)
        out += residual

        return out
