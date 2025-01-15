import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch.optim as optim
from torchvision.transforms import functional as F


class CustomObjectDetectionDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_filenames = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load image
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.images_dir, image_filename)
        image = Image.open(image_path).convert("RGB")
        
        # Load corresponding label file
        label_filename = os.path.splitext(image_filename)[0] + ".txt"  # assuming same name for label and image
        label_path = os.path.join(self.labels_dir, label_filename)
        
        # Read the label file
        boxes = []
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                label, x_center, y_center, width, height = map(float, line.strip().split())
                labels.append(int(label))  # Class label (0 or 1)
                
                # Convert normalized bounding box to (x_min, y_min, x_max, y_max)
                x_min = x_center - width / 2
                y_min = y_center - height / 2
                x_max = x_center + width / 2
                y_max = y_center + height / 2
                boxes.append([x_min, y_min, x_max, y_max])

        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels
        }
        
        # Apply image transformations if any
        if self.transform:
            image = self.transform(image)

        return image, target

# Define any transformations (e.g., resize, normalize)
transform = T.Compose([
    T.Resize((416, 416)),  # Resize to 416x416 for YOLO-style model
    T.ToTensor()  # Convert image to PyTorch tensor
])

# Initialize the dataset and data loader
images_dir = "./data/yolo_data_v1.yolov8/train/images"
labels_dir = "./data/yolo_data_v1.yolov8/train/labels"
dataset = CustomObjectDetectionDataset(images_dir, labels_dir, transform=transform)

# Create data loader
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Example of iterating through the data loader
# for images, targets in data_loader:
#     print(images)   # List of image tensors
#     print(targets)  # List of target dictionaries with 'boxes' and 'labels'


model = fasterrcnn_resnet50_fpn(pretrained=True)
model.train()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

for images, targets in data_loader:
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

