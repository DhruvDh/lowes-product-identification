import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import sys
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cam = cv2.VideoCapture(0)
model = 'squeezenet'
# model = 'inceptionv3'


def _find_classes(dir):
    """
    Finds the class folders in a dataset.

    Args:
        dir (string): Root directory path.

    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

    Ensures:
        No class is a subdirectory of another.
    """
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(
            dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


classes, class_to_idx = _find_classes('../data/combined')
idx_to_classes = {v: k for k, v in class_to_idx.items()}

label_names = classes

label_names = [name.replace('_', ' ') for name in label_names]
print(label_names)

mirror = True

cam.set(3, 1920)  # Set width
cam.set(4, 1080)  # Set height
cam.set(5, 60)    # Set Frame Rate

# |---- Code for drawing ----|
# |---- Code for drawing ----|
# |---- Code for drawing ----|

font = cv2.FONT_HERSHEY_SIMPLEX
bottom_center = (int(cam.get(3) / 2) - 270, int(cam.get(4) / 2) + 490)
top_center = (int(cam.get(3) / 2) - 110, int(cam.get(4) / 2) - 470)
fontScale = 1.25
light_blue = (229, 182, 21)
line_type = 2

cv2.namedWindow('lowes-demo', cv2.WINDOW_NORMAL)

x1 = int((cam.get(3) / 2) - 330)
y1 = int(100)
x2 = int((cam.get(3) / 2) + 330)
y2 = int(cam.get(4) - 100)
print("Rectangle (x1, y1): (", x1, ',', y1, ')')
print("--------- (x2, y2): (", x2, ',', y2, ')')


#  |----Code for Model----|
#  |----Code for Model----|
#  |----Code for Model----|

if model == 'inceptionv3':
    squeezenet = torchvision.models.inception_v3()
    squeezenet.num_classes = len(label_names)
    squeezenet.fc = nn.Linear(2048, squeezenet.num_classes)
    squeezenet.AuxLogits.fc = nn.Linear(768, squeezenet.num_classes)
    model_ = torch.load('../models/_files/inceptionv3_full')
    squeezenet.load_state_dict(model_['state_dict'])

    predict = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop((abs(y1-y2), abs(x1-x2))),
        torchvision.transforms.Resize((299, 299)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
        torchvision.transforms.Lambda(
            lambda x: squeezenet(x.unsqueeze(0).half().to(device))),
        torchvision.transforms.Lambda(
            lambda x: torch.argmax(x, dim=1).item()),
    ])
else:
    squeezenet = torchvision.models.squeezenet1_1()
    squeezenet.num_classes = len(label_names)
    layers = [
        nn.Dropout(0.5),
        nn.Conv2d(512, squeezenet.num_classes,
                  kernel_size=(1, 1), stride=(1, 1)),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=13, stride=1, padding=0)
    ]
    squeezenet.classifier = nn.Sequential(*layers)
    model_ = torch.load('../models/_files/squeeznet_full')
    squeezenet.load_state_dict(model_['state_dict'])

    predict = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop((abs(y1-y2), abs(x1-x2))),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
        torchvision.transforms.Lambda(
            lambda x: squeezenet(x.unsqueeze(0).half().to(device))),
        torchvision.transforms.Lambda(
            lambda x: torch.argmax(x, dim=1).item()),
    ])

squeezenet.half()
squeezenet.eval()
squeezenet.to(device)


def overlay_transparent(bg_img, img_to_overlay_t, x, y):
    b, g, r, a = cv2.split(img_to_overlay_t)
    overlay_color = cv2.merge((b, g, r))

    h, w, _ = overlay_color.shape
    roi = bg_img[y:y+h, x:x+w]

    # Black-out the area behind the logo in our original ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(a))

    # Mask out the logo from the logo image.
    img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=a)

    # Update the original image with our new ROI
    bg_img[y:y+h, x:x+w] = cv2.add(img1_bg, img2_fg)

    return bg_img


overlay = cv2.resize(cv2.imread('../overlay.png', -1), (1920, 1080))

product_imgs = [cv2.resize(cv2.imread(
    '_files/img/'+x+'.png', -1), (375, 360)) for x in classes]

while True:
    ret_val, img = cam.read()
    img = cv2.flip(img, 1)
    pred = predict(Image.fromarray(img))
    label = label_names[pred]
    cv2.putText(img, model,
                top_center,
                font,
                fontScale,
                light_blue,
                line_type)

    cv2.rectangle(img, (x1, y1), (x2, y2), light_blue, 3)

    cv2.putText(img, label,
                bottom_center,
                font,
                fontScale,
                light_blue,
                line_type)

    if pred == 10:
        cv2.imshow('lowes-demo', overlay_transparent(img, overlay, 0, 0))
    else:
        cv2.imshow('lowes-demo', overlay_transparent(overlay_transparent(
            img, overlay, 0, 0), product_imgs[pred], 1400, 360))

    if cv2.waitKey(1) == 27:
        break  # esc to quit
        cv2.destroyAllWindows()

cv2.destroyAllWindows()
