from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model.VanilaCNN import VanilaCNN
import torch.nn as nn
import torch.optim as optim
import torch
import os
import matplotlib.pyplot as plt

'''

__init__ 에 optimizer 에 따른 if 문
'''


class Trainer:
    def __init__(self, model):
        self.model = model
        self.set_hyperparameter()
        self.set_optimizer()
        self.set_loss()

    def set_model(self, model):
        self.model = model

    def set_optimizer(self, optimizer_name='SGD'):
        if optimizer_name == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif optimizer_name == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

    def set_loss(self, loss_name='XEntropy'):
        if loss_name == 'XEntropy':
            self.loss_func = nn.CrossEntropyLoss()
        elif loss_name == 'MSE':
            self.loss_func = nn.MSELoss()

    def set_hyperparameter(self, lr=0.01, batch_size=5, epoch=2):
        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epoch

    def train(self):
        train_dir = './preprocessed_data/train'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        train_data = datasets.ImageFolder(train_dir, transform=transforms.Compose(
            [transforms.Resize(255),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
             ]))  # normalize?
        train_loader = DataLoader(train_data, batch_size=self.batch_size)

        loss_arr = []
        for i in range(self.epoch):
            for j, [image, label] in enumerate(train_loader):
                x = image.to(device)
                y_ = label.to(device)

                self.optimizer.zero_grad()
                output = self.model.forward(x)
                loss = self.loss_func(output, y_)
                loss_arr.append(loss.cpu().detach().numpy())
                loss.backward()
                self.optimizer.step()

                print(loss)
                if j == 50:
                    break




def train_models():
    vanila_path = './model/vanila.pt'

    vanila_model = VanilaCNN()

    model_trainer = Trainer(vanila_model)

    if not os.path.exists(vanila_path):
        model_trainer.set_model(vanila_model)
        model_trainer.train()
       # torch.save(vanila_model, vanila_path)


if __name__ == '__main__':
    train_models()