import torch
import cv2
import numpy as np
import torchvision.transforms as transforms

from detector import face_detector


def predict_res18(img):
    res18_path = './trained_model/res18.pt'

    res18_model = torch.load(res18_path)
    res18_model.eval()

    img_list = face_detector(img)

    img_list=torch.Tensor(img_list)
    print(img_list.shape)

    transform_img = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Resize(255),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 의미?
         ]
    )

    prediction = torch.argmax(res18_model(transform_img(img).float()), dim=1)

    #for (x, y, w, h), age in zip(faces, prediction):
        #img = cv2.rectangle(img, pt1=(x - int(w / 2), y + int(h / 2)), pt2=(x + int(w / 2), y - int(h / 2)))

    cv2.imshow(img)
    cv2.waitKey(0)


def predict_all():
    img = cv2.imread('./out/face_ex1.jpg')
    predict_res18(img)


if __name__ == '__main__':
    predict_all()
