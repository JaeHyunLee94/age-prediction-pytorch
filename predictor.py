import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from detector import face_detector

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
age_tensor = torch.tensor([i for i in range(1, 101)]).type(torch.FloatTensor).to(device)


def image_loader(image):
    transform_img = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Resize(255),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 의미?
         ]
    )
    """load image, returns cuda tensor"""
    image = np.array(image)
    image = transform_img(image).float()
    image = image.unsqueeze(0)
    return image.to(device)  # assumes that you're using CPU


def predict_res18(img):
    res18_path = './trained_model/res18.pt'

    res18_model = torch.load(res18_path)
    res18_model.eval()

    faces, img_list = face_detector(img)
    if not img_list:
        return img

    output = res18_model(image_loader(img_list[0]))
    output = F.softmax(output, dim=1)
    output = (output * age_tensor).sum(dim=1)

    print(output)
    for (x, y, w, h), age in zip(faces, output):
        img = cv2.rectangle(img, pt1=(x, y), pt2=(x + int(w), y + int(h)), color=(255, 0, 0))
        cv2.putText(img, text='Age: {:.3f}'.format(age.item(), 3), org=(x, y), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=0.3,
                    color=(0, 255, 0))

    return img


def predict_all():
    img = cv2.imread('./out/face_ex3.jpeg')
    cv2.imshow('sdf', predict_res18(img))
    cv2.waitKey(0)
    cap = cv2.VideoCapture(0)

    if cap.isOpened():
        print('webcap had opened')

    while True:
        ret, frame = cap.read()

        if ret:
            if cv2.waitKey(10) & 0XFF == ord('q'):
                break
            cv2.imshow('video', predict_res18(frame))

        else:
            print('error')

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    predict_all()
