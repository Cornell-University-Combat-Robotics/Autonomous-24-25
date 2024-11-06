import torch
import cv2
from torchvision import transforms
import numpy as np
def predict(img):
    # for some reason last year we had the imports in this function?
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])
    img = transform(img)

    # img = np.expand_dims(img, 0)
    # img = torch.from_numpy(img)

    model = torch.load("TODO: MODEL FILEPATH")
    cnn.eval()

    labels, boxes = model(img)
    # TODO: Figure out what the model returns and how to interpret it
    

if __name__ == 'main':
    