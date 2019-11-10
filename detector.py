import cv2


def face_detector(img):
    face_cascade = cv2.CascadeClassifier()
    face_cascade.load('./out/haarcascade_frontalface_default.xml')
    imgNum = 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    img_list=[]
    for (x, y, w, h) in faces:
        cropped = img[y - int(h / 4):y + h + int(h / 4), x - int(w / 4):x + w + int(w / 4)]
        img_list.append(cropped)

    return img_list

