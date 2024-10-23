

import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import yaml

import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Define the model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.train()

# Example dataset loader (custom dataset should be used here)
# This returns images and target dictionaries that include bounding boxes and labels
def collate_fn(batch):
    return tuple(zip(*batch))

# THIS COULD GO POORLY ---------------------------
# print("opening yaml")
with open('./data/yolo_data_v1.yolov8/nhrl_bots.yaml','r') as file: #edit yaml path
    config = yaml.safe_load(file)
# print("finished reading yaml")

train_images_dir = config['train']
train_labels_dir = train_images_dir.replace('images', 'labels')
# print("found train dir")
val_images_dir = config['val']
val_labels_dir = val_images_dir.replace('images', 'labels')
# print("found val dir")
test_images_dir = config['test']
test_labels_dir = test_images_dir.replace('images', 'labels')
# print("found test dir")

class Data(Dataset):
    """
    A custom PyTorch Dataset class for loading images and corresponding YOLO-format labels.
    """
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_paths = [os.path.join(self.image_dir, img) for img in os.listdir(self.image_dir) if img.endswith('.jpg')]
    
    
    def load_yolo_labels(self, label_path):
        """
        Reads a label file and returns a list of tuples with class index and bounding box values.
        
        Returns:
        list of tuples: Each tuple contains (class_index, x_center, y_center, width, height).
        """
        boxes = []
        labels = []

        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                class_index = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])

                x_min = x_center - width / 2
                y_min = y_center - height / 2
                x_max = x_center + width / 2
                y_max = y_center + height / 2

                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(class_index)

        return boxes, labels
        
        
    def __getitem__(self,idx):
        """
        Retrieves and processes an image and its corresponding labels from the dataset.
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
        boxes, labels = self.load_yolo_labels(label_path)
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        return image, {"boxes": boxes, "labels": labels}
    
    def __len__(self):
        return len(self.image_dir)
    
transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])

train_dataset = Data(train_images_dir, train_labels_dir, transform=transform)
val_dataset = Data(val_images_dir, val_labels_dir, transform=transform)

training_loader = DataLoader(train_dataset, batch_size = 4, shuffle=True)
validation_loader = DataLoader(val_dataset, batch_size = 4, shuffle=False)

optimizer = optim.Adam(model.parameters(), lr=0.001)
# END THIS COULD GO POORLY -----------------------

# Assume data_loader provides batches of (images, targets)
for images, targets in training_loader:
    images = [F.to_tensor(img) for img in images]  # Convert images to tensors

    # Example target: [{"boxes": [[x1, y1, x2, y2], ...], "labels": [1, 2, ...]}, ...]
    targets = [{"boxes": torch.tensor(target["boxes"]), "labels": torch.tensor(target["labels"])} for target in targets]

    # Forward pass
    loss_dict = model(images, targets)

    # Total loss
    losses = sum(loss for loss in loss_dict.values())

    # Backpropagation and optimization steps
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()
