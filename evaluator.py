# model 이랑 img 넣으면 나이 추청

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

test_dir = './preprocessed_data/test'
vanila_path = './model/vanila.pt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Evaluator:
    def __init__(self):
        pass

    def set_model(self, model):
        self.model = model

    def evaluate(self):
        pass


def evaluate_models():
    vanila_model = torch.load(vanila_path)

    test_data = datasets.ImageFolder(test_dir, transform=transforms.Compose(
        [transforms.Resize(255),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         ]))  # normalize?
    test_loader = DataLoader(test_data)


if __name__ == '__main__':
    evaluate_models()
