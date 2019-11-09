from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch
import torch.nn as nn

'''

__init__ 에 optimizer 에 따른 if 문
'''


class Trainer:
    def __init__(self, model, lr=0.01, batch_size=200, epoch=5, optimizer='Adam', loss='XEntropy'):
        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epoch
        self.optimizer = optimizer
        self.loss = loss

    def set_model(self, model):
        self.model = model

    def train(self):
        train_dir = './preprocessed_data/train'

        train_data = datasets.ImageFolder(train_dir, transform=transforms.Compose(
            [transforms.Resize(255),
             transforms.CenterCrop(224),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
             transforms.ToTensor()]))  # normalize?
        train_loader = DataLoader(train_data, batch_size=self.batch_size)

        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for i in range(self.epoch):
            for j, [image, label] in enumerate(train_loader):
                x = image
                y_ = label

                optimizer.zero_grad()
                output = self.model.forward(x)
                loss = loss_func(output, y_)
                loss.backward()
                optimizer.step()

                if j % 5 == 0:
                    print(loss)
        self.model.is_trained = True
