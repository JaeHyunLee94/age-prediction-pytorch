import torch
import torch.nn as nn

'''
batch_norm 넣기
'''


class VanilaCNN(nn.Module):
    def __init__(self):
        super(VanilaCNN, self).__init__()
        self.cnn_layer = nn.Sequential(

            nn.Conv2d(3, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(173056, 50),
            nn.ReLU(),
            nn.Linear(50, 100)
        )

    def forward(self, x):
        out = self.cnn_layer(x)
        out = out.view(out.size(0), -1)
        out = self.fc_layer(out)
        return out
