import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from model import DenseNet as DenseNet

test_dir = './preprocessed_data/test'
vanila_path = './trained_model/vanila.pt'
res18_path = './trained_model/res18.pt'

squeeze1_0_path = './trained_model/squeeze1_0.pt'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
age_tensor = torch.tensor([i for i in range(1, 101)]).type(torch.FloatTensor).to(device)


def evaluate(model, batch_size=128):
    # writer = SummaryWriter()  # logdir 설정
    loss_func = nn.L1Loss()
    model.eval()
    model.to(device)

    test_data = datasets.ImageFolder(test_dir, transform=transforms.Compose(
        [transforms.Resize(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         ]))  # normalize?
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    test_iter = 0
    test_list = []
    with torch.no_grad():
        for i, [image, label] in enumerate(test_loader):
            x = image.to(device)
            y_ = label.type(torch.FloatTensor).to(device)

            output = model.forward(x)
            output = F.softmax(output, dim=1)
            output = (output * age_tensor).sum(dim=1)
            test_loss = loss_func(output, y_)
            test_list.append(test_loss.item())
            print('test_loss: ', test_loss)

            # Tensorboard : test_loss
            # writer.add_scalar('Loss/test', test_loss.item(), test_iter)
            # test_iter += 1
    print('total test loss: ', sum(test_list) / len(test_list))


def evaluate_models():
    dense121 = DenseNet.get_densenet121()

    dense121.load_state_dict(torch.load('./trained_model/densenet121.pt'))

    evaluate(model=dense121, batch_size=36)


if __name__ == '__main__':
    evaluate_models()
