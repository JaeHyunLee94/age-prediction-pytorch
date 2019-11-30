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
from sampler import ImbalancedDatasetSampler

'''
__init__ 에 optimizer 에 따른 if 문
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_dir = './preprocessed_data/train'
validation_dir = './preprocessed_data/validate'
age_tensor = torch.tensor([i for i in range(1, 101)]).type(torch.FloatTensor).to(device)

data_transforms = {
    'train': transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 의미?
                                 ]),
    'val': transforms.Compose([transforms.Resize(255),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 의미?
                               ]),
}


class Trainer:
    def __init__(self, model):
        self.model = model
        self.set_hyperparameter()
        self.set_optimizer()
        self.set_loss()
        self.set_lr_schedule()

    def set_model(self, model):
        self.model = model

    def set_optimizer(self, optimizer_name='Adam'):
        if optimizer_name == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_dacay)
        elif optimizer_name == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_dacay)
        elif optimizer_name == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr, weight_decay=self.weight_dacay)
        elif optimizer_name == 'AdaGrad':
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.lr, weight_decay=self.weight_dacay)
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

    def set_hyperparameter(self, lr=0.0001, batch_size=128, epoch=60, weight_decay=0):  # 조정!: 128,50?
        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epoch
        self.weight_dacay = weight_decay

    def set_lr_schedule(self, step_size=10, gamma=0.1):
        self.scheculer = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    def train(self):

        self.model.to(device)

        train_data = datasets.ImageFolder(train_dir, transform=data_transforms['train'])

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        writer = SummaryWriter()
        iter = 0

        val_loss = 999
        threshold = 0

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
                    100 * (self.batch_size * j + 20000 * i) / (self.epoch * 20000)))

            writer.add_scalar('Loss/train', sum(train_loss) / len(train_loss), iter)
            iter += 1
            self.scheculer.step()
            val_loss_next = self.validate()
            writer.add_scalar('Loss/Validation', val_loss_next, iter)
            iter += 1

            if val_loss_next > val_loss:
                print('Early stopping')
                break
            elif val_loss - val_loss_next < 0.05:
                threshold += 1
            else:
                threshold = 0

            val_loss = val_loss_next

            print(threshold)

            if threshold >= 2:
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

    densenet121_model = DenseNet.get_densenet121()
    model_trainer = Trainer(densenet121_model)
    model_trainer.set_hyperparameter(batch_size=36, lr=0.0015)
    model_trainer.set_lr_schedule(5, 0.34)
    model_trainer.train()
    torch.save(densenet121_model.state_dict(), densenet121_path)
    del densenet121_model, model_trainer

    densenet121_model = DenseNet.get_densenet121()
    model_trainer = Trainer(densenet121_model)
    model_trainer.set_hyperparameter(batch_size=36, lr=0.0015)
    model_trainer.set_lr_schedule(3, 0.45)
    model_trainer.train()
    torch.save(densenet121_model.state_dict(), densenet121_path)
    del densenet121_model, model_trainer

    densenet161_model = DenseNet.get_densenet161()
    model_trainer = Trainer(densenet161_model)
    model_trainer.set_hyperparameter(batch_size=20, lr=0.0005)
    model_trainer.set_lr_schedule(3, 0.6)
    model_trainer.train()
    torch.save(densenet161_model.state_dict(), densenet161_path)
    del densenet161_model, model_trainer


if __name__ == '__main__':
    train_models()
