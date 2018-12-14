import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
label_names = os.listdir('../data/')

label_names = [name.replace('_', ' ') for name in label_names]

mirror = True
cam = cv2.VideoCapture(1)
cam.set(3, 1280)
cam.set(4, 720)
cam.set(5, 60)
print(cam.get(8))
print(cam.get(18))

squeezenet = torchvision.models.squeezenet1_1()
squeezenet.num_classes = len(label_names)

layers = [
    nn.Dropout(0.5),
    nn.Conv2d(512, squeezenet.num_classes, kernel_size=(1, 1), stride=(1, 1)),
    nn.ReLU(inplace=True),
    nn.AvgPool2d(kernel_size=13, stride=1, padding=0),
    nn.Softmax()
]

squeezenet.classifier = nn.Sequential(*layers)
predict = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop((448, 448)),
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
    torchvision.transforms.Lambda(
        lambda x: squeezenet(x.unsqueeze(0).half().to(device))),
    torchvision.transforms.Lambda(
        lambda x: torch.argmax(x, dim=1).item()),
])

model = torch.load('squeeznet_half')
squeezenet.load_state_dict(model['state_dict'])
squeezenet.half()
squeezenet.eval()
squeezenet.to(device)

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (int(cam.get(3) / 2) - 274, int(cam.get(4) / 2) - 274)
fontScale = 1
fontColor = (0, 255, 0)
lineType = 2

while True:
    ret_val, img = cam.read()
    if mirror:
        img = cv2.flip(img, 1)

    cv2.putText(img, label_names[predict(Image.fromarray(img))],
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    x1 = int((cam.get(3) / 2) - 224)
    x2 = int((cam.get(3) / 2) + 224)
    y1 = int((cam.get(4) / 2) - 224)
    y2 = int((cam.get(4) / 2) + 224)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv2.imshow('demo', img)

    if cv2.waitKey(1) == ord('f'):
        cam = cv2.VideoCapture(0)
        cam.set(3, 1280)
        cam.set(4, 720)
        cam.set(5, 60)
    if cv2.waitKey(1) == ord('b'):
        cam = cv2.VideoCapture(1)
        cam.set(3, 1280)
        cam.set(4, 720)
        cam.set(5, 60)
    if cv2.waitKey(1) == 27:
        break  # esc to quit
        cv2.destroyAllWindows()

cv2.destroyAllWindows()
