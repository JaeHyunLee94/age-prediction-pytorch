import torch.nn as nn

'''
batch norm?
'''


class VanilaCNN(nn.Module):
    def __init__(self):
        super(VanilaCNN, self).__init__()
        self.cnn_layer = nn.Sequential(

            nn.Conv2d(3, 16, 5),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(173056, 100),
            nn.ReLU(),
            nn.Linear(100, 100)
        )

    def forward(self, x):
        out = self.cnn_layer(x)
        out = out.view(out.size(0), -1)
        out = self.fc_layer(out)
        return out


def get_vanila():
    model=VanilaCNN()
    return model