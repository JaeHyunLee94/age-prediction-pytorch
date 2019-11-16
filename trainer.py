from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import model.VanilaCNN as VanilaCNN
import model.VGG as VGG
import model.Resnet as Resnet
import model.Inception as inception
import model.SqueezeNet as SqueezeNet
import  model.DenseNet as DenseNet
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
import torch

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

    def set_hyperparameter(self, lr=0.005, batch_size=128, epoch=1):  # 조정!: 128,50?
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
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)

        writer = SummaryWriter()
        train_iter = 0
        total = 0
        correct = 0
        for i in range(self.epoch):

            for j, [image, label] in enumerate(train_loader, 1):

                x = image.to(device)
                y_ = label.to(device)

                self.optimizer.zero_grad()
                output = self.model.forward(x)

                total += y_.size(0)
                correct += (torch.abs(torch.argmax(output, dim=1) - y_) <= 5).sum().float()
                accuracy = (correct / total) * 100
                print('batch training Accuracy: ', accuracy)

                loss = self.loss_func(output, y_)
                loss.backward()
                self.optimizer.step()

                print('Real time loss: ', loss)

                if j % 100 == 0:  # 언제 print?
                    writer.add_scalar('Loss/train', loss.item(), train_iter)
                    writer.add_scalar('Accuracy/train', accuracy.item(), train_iter)
                    train_iter += 1

                # if j == 10:
                # break


def train_models():
    vanila_path = './trained_model/vanila.pt'

    res6_path = './trained_model/res4.pt'
    res12_path = './trained_model/res7.pt'
    res18_path = './trained_model/res18.pt'
    res34_path = './trained_model/res34.pt'

    squeeze1_0_path = './trained_model/squeeze1_0.pt'
    squeeze1_1_path = './trained_model/squeeze1_1.pt'

    densenet121= './trained_model/densenet121.pt'
    densenet161= './trained_model/densenet161.pt'

    vgg11_path = './trained_model/vgg11.pt'

    inceptionv3_path = './trained_model/inceptionv3.pt'


    res18_model = torch.load(res18_path)

    model_trainer = Trainer(res18_model)

    #if not os.path.exists(res18_path):
    model_trainer.set_model(res18_model)
    model_trainer.train()
    torch.save(res18_model, res18_path)


if __name__ == '__main__':
    train_models()
