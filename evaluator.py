# model 이랑 img 넣으면 나이 추청

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

test_dir = './preprocessed_data/test'
vanila_path = './model/vanila.pt'
res18_path = './model/res18.pt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate(model, batch_size=128):
    writer = SummaryWriter()
    loss_func = nn.CrossEntropyLoss() # model.lossfunc
    model.eval()

    test_data = datasets.ImageFolder(test_dir, transform=transforms.Compose(
        [transforms.Resize(255),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         ]))  # normalize?
    test_loader = DataLoader(test_data, batch_size=batch_size)

    total = 0
    correct = 0
    test_iter = 0
    test_acc_iter = 0
    with torch.no_grad():
        for i, [image, label] in enumerate(test_loader):
            x = image.to(device)
            y_ = label.to(device)

            output = model.forward(x)
            _, output_index = torch.max(output, 1)

            test_loss = loss_func(output, y_)

            total += label.size(0)
            correct += (torch.abs(output_index - y_) <= 5).sum().float()

            accuracy = 100 * correct / total

            print('test accuracy: ',accuracy)

            # Tensorboard : test_loss
            writer.add_scalar('Loss/test', test_loss.item(), test_iter)
            test_iter += 1

            # Tensorboard : test_Accuracy
            writer.add_scalar('Accuracy/test', accuracy.item(), test_acc_iter)
            test_acc_iter += 1

        print("Accuracy of Test Data : {}".format(100 * correct / total))


def evaluate_models():
    # vanila_model = torch.load(vanila_path)
    res18_model = torch.load(res18_path)
    evaluate(model=res18_model, batch_size=128)


if __name__ == '__main__':
    evaluate_models()
