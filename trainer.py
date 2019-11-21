from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import model.VanilaCNN as VanilaCNN
import model.VGG as VGG
import model.Resnet as Resnet
import model.Inception as inception
import model.SqueezeNet as SqueezeNet
import model.DenseNet as DenseNet
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
import os

'''
__init__ 에 optimizer 에 따른 if 문
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_dir = './preprocessed_data/train'
validation_dir = './preprocessed_data/validation'
age_tensor = torch.tensor([i for i in range(1, 101)]).type(torch.FloatTensor).to(device)


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
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_dacay)

    def set_loss(self, loss_name='L1Loss'):
        if loss_name == 'XEntropy':
            self.loss_func = nn.CrossEntropyLoss()
        elif loss_name == 'MSE':
            self.loss_func = nn.MSELoss(reduction='mean')
        elif loss_name == 'L1Loss':
            self.loss_func = nn.L1Loss(reduction='mean')

    def set_hyperparameter(self, lr=0.005, batch_size=128, epoch=15, weight_decay=0):  # 조정!: 128,50?
        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epoch
        self.weight_dacay = weight_decay

    def train(self):

        self.model.to(device)

        train_data = datasets.ImageFolder(train_dir, transform=transforms.Compose(
            [transforms.Resize(255),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 의미?
             ]))

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)

        writer = SummaryWriter()
        train_iter = 0
        batch_iter = 0
        val_acc = 0
        val_loss = 999
        threshold = 0

        for i in range(self.epoch):
            self.model.train()
            for j, [image, label] in enumerate(train_loader, 1):

                x = image.to(device)
                y_ = label.type(torch.FloatTensor).to(device)

                total = 0
                correct = 0

                self.optimizer.zero_grad()
                output = self.model.forward(x)
                output = F.softmax(output, dim=1)
                output = (output * age_tensor).sum(dim=1)
                loss = self.loss_func(output, y_)
                loss.backward()
                self.optimizer.step()

                total += y_.size(0)

                correct += ((output - y_ <= 5) * (output - y_ >= -5)).sum().float()

                accuracy = (correct / total) * 100

                print('batch training Accuracy: ', accuracy)
                print('Real time loss: ', loss)
                print('Training Percesnt : --------{}%--------'.format(
                    100 * (self.batch_size * j + 93822 * i) / (self.epoch * 93822)))

                if j % 50 == 0:  # 언제 print?
                    writer.add_scalar('Loss/train', loss.item(), train_iter)
                    writer.add_scalar('Accuracy/train', accuracy.item(), train_iter)
                    train_iter += 1

            val_acc_next, val_loss_next = self.validate()

            if val_acc_next < val_acc:
                threshold += 1
            else:
                threshold = 0

            val_acc = val_acc_next
            val_loss = val_loss_next

            print(threshold)

            if threshold > 2:
                break

    def validate(self):

        self.model.eval()

        validation_data = datasets.ImageFolder(validation_dir, transform=transforms.Compose(
            [transforms.Resize(255),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 의미?
             ]))  # validation 에서도 normalize?

        validation_loader = DataLoader(validation_data, batch_size=self.batch_size, shuffle=True)

        total = 0
        correct = 0
        val_iter = 0

        val_loss = 0
        with torch.no_grad():
            for i, [image, label] in enumerate(validation_loader):
                val_iter += 1
                x = image.to(device)
                y_ = label.type(torch.FloatTensor).to(device)

                output = self.model.forward(x)
                output = F.softmax(output, dim=1)
                output = (output * age_tensor).sum(dim=1)

                val_loss += self.loss_func(output, y_)

                total += y_.size(0)

                correct += ((output - y_ <= 5) * (output - y_ >= -5)).sum().float()

                if i % 5 == 0:
                    print('Validating model')

        accuracy = 100 * correct / total
        val_loss_avg = val_loss / val_iter
        print('validation accuracy: ', accuracy)
        print('validaion loss: ', val_loss_avg)
        return accuracy, val_loss_avg


def train_models():
    vanila_path = './trained_model/vanila.pt'

    res6_path = './trained_model/res6.pt'
    res12_path = './trained_model/res7.pt'
    res18_path = './trained_model/res18.pt'
    res34_path = './trained_model/res34.pt'

    squeeze1_0_path = './trained_model/squeeze1_0.pt'
    squeeze1_1_path = './trained_model/squeeze1_1.pt'

    densenet121 = './trained_model/densenet121.pt'
    densenet161 = './trained_model/densenet161.pt'

    vgg11_path = './trained_model/vgg11.pt'

    inceptionv3_path = './trained_model/inceptionv3.pt'

    res18_model = torch.load(res18_path)

    model_trainer = Trainer(res18_model)

    #if not os.path.exists(res18_path):
        # model_trainer.set_model(res6_model)
    model_trainer.train()
    torch.save(res18_model, res18_path)


if __name__ == '__main__':
    train_models()
