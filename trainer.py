from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model.VanilaCNN import VanilaCNN
import model.Resnet as Resnet
from torch.utils.tensorboard import SummaryWriter
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

    def set_optimizer(self, optimizer_name='Adam'):
        if optimizer_name == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif optimizer_name == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

    def set_loss(self, loss_name='XEntropy'):
        if loss_name == 'XEntropy':
            self.loss_func = nn.CrossEntropyLoss()
        elif loss_name == 'MSE':
            self.loss_func = nn.MSELoss()

    def set_hyperparameter(self, lr=0.01, batch_size=20, epoch=5):  # 조정!: 128,50?
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
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 의미?
             ]))
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        loss_arr = []
        accuracy_arr = []

        writer = SummaryWriter()
        train_iter = 0
        for i in range(self.epoch):
            for j, [image, label] in enumerate(train_loader):

                total = 0
                correct = 0

                x = image.to(device)
                y_ = label.to(device)

                self.optimizer.zero_grad()
                output = self.model.forward(x)
                total += y_.size(0)
                correct += (torch.abs(torch.argmax(output, dim=1) - y_) <= 5).sum().float()
                accuracy = (correct / total) * 100
                print('Real time Accuracy: ', accuracy)
                accuracy_arr.append(accuracy)
                loss = self.loss_func(output, y_)
                loss_arr.append(loss.cpu().detach().numpy())
                loss.backward()
                self.optimizer.step()

                print('Real time loss: ', loss)
                if j % 5 == 0:  # 언제 print?
                    writer.add_scalar('Loss/train', loss.item(), train_iter)
                    writer.add_scalar('Accuracy/train', accuracy.item(), train_iter)
                    train_iter += 1
                if j == 1000:
                    break
        print(accuracy_arr)
        plt.plot(accuracy_arr)
        plt.title('trian_accuracy')
        plt.xlabel('1 batch')
        plt.ylabel('train_accuracy')
        plt.savefig('./out/train_acc.png')
        plt.close()


def train_models():
    vanila_path = './model/vanila.pt'
    res4_path = './model/res4.pt'
    res7_path = './model/res7.pt'

    vanila_model = VanilaCNN()
    res4_model = Resnet.resnet4()
    res7_model = Resnet.resnet7()

    model_trainer = Trainer(vanila_model)

    if not os.path.exists(vanila_path):
        pass
        model_trainer.set_model(vanila_model)
        model_trainer.train()
        torch.save(vanila_model, vanila_path)

    if not os.path.exists(res4_path):
        pass
        model_trainer.set_model(res4_model)
        model_trainer.train()
        torch.save(res4_model, res4_path)

    if not os.path.exists(res7_path):
        model_trainer.set_model(res7_model)
        model_trainer.train()
        torch.save(res7_model, res7_path)


if __name__ == '__main__':
    train_models()
