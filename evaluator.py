# model 이랑 img 넣으면 나이 추청

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

test_dir = './preprocessed_data/test'
vanila_path = './model/vanila.pt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate(self, model, batch_size):
    writer = SummaryWriter()

    test_data = datasets.ImageFolder(test_dir, transform=transforms.Compose(
        [transforms.Resize(255),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         ]))  # normalize?
    test_loader = DataLoader(test_data, batch_size=batch_size)
    with torch.no_grad():
        for i, [image, label] in enumerate(test_loader):
            x = image.to(device)
            y_ = label.to(device)


def evaluate_models():
    vanila_model = torch.load(vanila_path)



if __name__ == '__main__':
    evaluate_models()
