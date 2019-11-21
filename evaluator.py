# model 이랑 img 넣으면 나이 추청

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

test_dir = './preprocessed_data/test'
vanila_path = './trained_model/vanila.pt'
res18_path = './trained_model/res18.pt'

squeeze1_0_path = './trained_model/squeeze1_0.pt'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
age_tensor = torch.tensor([i for i in range(1, 101)]).type(torch.FloatTensor).to(device)


def evaluate(model, batch_size=128):
    writer = SummaryWriter()
    loss_func = nn.L1Loss()
    model.eval()

    test_data = datasets.ImageFolder(test_dir, transform=transforms.Compose(
        [transforms.Resize(255),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         ]))  # normalize?
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    total = 0
    correct = 0
    test_iter = 0
    test_acc_iter = 0
    with torch.no_grad():
        for i, [image, label] in enumerate(test_loader):
            x = image.to(device)
            y_ = label.type(torch.FloatTensor).to(device)

            output = model.forward(x)
            output = F.softmax(output, dim=1)
            output = (output * age_tensor).sum(dim=1)
            test_loss = loss_func(output, y_)

            total += y_.size(0)

            correct += ((output - y_ <= 5) * (output - y_ >= -5)).sum().float()

            accuracy = (correct / total) * 100

            print('test_loss: ', test_loss)
            print('test accuracy: ', accuracy)

            # Tensorboard : test_loss
            # writer.add_scalar('Loss/test', test_loss.item(), test_iter)
            # test_iter += 1

            # Tensorboard : test_Accuracy
            # writer.add_scalar('Accuracy/test', accuracy.item(), test_acc_iter)
            # test_acc_iter += 1

        print("Accuracy of Test Data : {}".format(100 * correct / total))


def evaluate_models():
    res18_model = torch.load('./trained_model/res18.pt')

    evaluate(model=res18_model, batch_size=128)


if __name__ == '__main__':
    evaluate_models()
