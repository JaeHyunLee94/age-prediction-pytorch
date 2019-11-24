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
<<<<<<< HEAD
age_tensor = torch.tensor([i for i in range(1,101)]).type(torch.FloatTensor).to(device)
=======
age_tensor = torch.tensor([i for i in range(70)]).type(torch.FloatTensor).to(device)
>>>>>>> 2eefc0787a2de19b8aefd6ba423b430da88f5f81


def evaluate(model, batch_size=128):
    writer = SummaryWriter() # logdir 설정
    loss_func = nn.L1Loss()
    model.eval()

    test_data = datasets.ImageFolder(test_dir, transform=transforms.Compose(
        [transforms.Resize(255),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         ]))  # normalize?
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    test_iter = 0
    with torch.no_grad():
        for i, [image, label] in enumerate(test_loader):
            x = image.to(device)
            y_ = label.type(torch.FloatTensor).to(device)

            output = model.forward(x)
            output = F.softmax(output, dim=1)
            output = (output * age_tensor).sum(dim=1)
            test_loss = loss_func(output, y_)

            print('test_loss: ', test_loss)

            # Tensorboard : test_loss
            writer.add_scalar('Loss/test', test_loss.item(), test_iter)
            test_iter += 1


def evaluate_models():
    res18_model = torch.load('./trained_model/res18.pt')

<<<<<<< HEAD
    evaluate(model=res18_model, batch_size=64)
=======
    evaluate(model=res18_model, batch_size=128)
>>>>>>> 2eefc0787a2de19b8aefd6ba423b430da88f5f81


if __name__ == '__main__':
    evaluate_models()
