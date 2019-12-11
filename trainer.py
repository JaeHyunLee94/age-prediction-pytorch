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

'''
__init__ 에 optimizer 에 따른 if 문
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_dir = './preprocessed_data/train'
validation_dir = './preprocessed_data/validate'
age_tensor = torch.tensor([i for i in range(1, 101)]).type(torch.FloatTensor).to(device)

data_transforms = {
    'train': transforms.Compose([transforms.Resize(224),
                                 #transforms.RandomHorizontalFlip(),
                                 #transforms.RandomRotation(10),
                                 #transforms.ColorJitter(0.4, 0.4, 0.4),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 의미?
                                 ]),
    'val': transforms.Compose([transforms.Resize(224),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 의미?
                               ]),
}


class Trainer:
    def __init__(self, model, optimizer_name='Adam', lr=0.0002, epoch=65, batch_size=256,
                 weight_decay=0.001, momentum=0.9,
                 nes=True, lr_decay=(50, 0.1)):
        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epoch
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nes = nes
        self.set_optimizer(optimizer_name)
        self.scheculer = optim.lr_scheduler.StepLR(self.optimizer, lr_decay[0], lr_decay[1])
        self.set_loss()

    def set_optimizer(self, optimizer_name):
        if optimizer_name == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif optimizer_name == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                       momentum=self.momentum, nesterov=self.nes)
        elif optimizer_name == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif optimizer_name == 'AdaGrad':
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif optimizer_name == 'Adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.lr, weight_decay=self.weight_dacay)
        else:
            raise ModuleNotFoundError

    def set_loss(self, loss_name='L1Loss'):
        if loss_name == 'XEntropy':
            self.loss_func = nn.CrossEntropyLoss()
        elif loss_name == 'MSE':
            self.loss_func = nn.MSELoss(reduction='mean')
        elif loss_name == 'L1Loss':
            self.loss_func = nn.L1Loss(reduction='mean')
        else:
            raise ModuleNotFoundError

    def train(self):

        self.model.to(device)

        train_data = datasets.ImageFolder(train_dir, transform=data_transforms['train'])

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        writer = SummaryWriter()
        iter = 0

        val_loss = 999
        threshold = 0
        patient = 0

        for i in range(self.epoch):
            self.model.train()
            train_loss = []
            for j, [image, label] in enumerate(train_loader, 1):
                x = image.to(device)
                y_ = label.type(torch.FloatTensor).to(device)
                self.optimizer.zero_grad()
                output = self.model.forward(x)
                output = F.softmax(output, dim=1)
                output = (output * age_tensor).sum(dim=1)
                loss = self.loss_func(output, y_)
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())
                print('Real time loss: ', loss)
                print('Training Percesnt : --------{}%--------'.format(
                    100 * (self.batch_size * j + 45000 * i) / (self.epoch * 45000)))
            iter += 1

            writer.add_scalar('Loss/train', sum(train_loss) / len(train_loss), iter)
            self.scheculer.step()
            val_loss_next = self.validate()
            writer.add_scalar('Loss/Validation', val_loss_next, iter)

            if val_loss_next > val_loss:
                patient += 1
            else:
                patient = 0
                if val_loss - val_loss_next < 0.01:
                    threshold += 1
                else:
                    threshold = 0

            val_loss = val_loss_next

            print(threshold)

            if threshold >= 15 or patient >= 5:
                break

    def validate(self):

        self.model.eval()

        validation_data = datasets.ImageFolder(validation_dir, transform=data_transforms['val'])

        validation_loader = DataLoader(validation_data, batch_size=self.batch_size, shuffle=True)

        total = 0
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

                val_loss += self.loss_func(output, y_).item()

                total += y_.size(0)

                if i % 5 == 0:
                    print('Validating model')

        val_loss_avg = val_loss / val_iter

        print('validaion loss: ', val_loss_avg)
        return val_loss_avg


def train_models():
    vanila_path = './trained_model/vanila.pt'

    res6_path = './trained_model/res6.pt'
    res12_path = './trained_model/res7.pt'
    res18_path = './trained_model/res18.pt'
    res34_path = './trained_model/res34.pt'

    squeeze1_0_path = './trained_model/squeeze1_0.pt'
    squeeze1_1_path = './trained_model/squeeze1_1.pt'

    densenet121_path = './trained_model/densenet121.pt'
    densenet161_path = './trained_model/densenet161.pt'

    vgg11_path = './trained_model/vgg11.pt'

    inceptionv3_path = './trained_model/inceptionv3.pt'

    resnet18_model = Resnet.get_resnet18()
    model_trainer = Trainer(resnet18_model)
    model_trainer.train()
    torch.save(resnet18_model.state_dict(), res18_path)
    del resnet18_model, model_trainer


if __name__ == '__main__':
    train_models()
