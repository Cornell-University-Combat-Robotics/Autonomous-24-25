import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import yaml

with open('/data/nhrl_bots.yaml','r') as file: #edit yaml path
    config = yaml.safe_load(file)

# set up all directory paths
train_images_dir = config['train']
train_labels_dir = train_images_dir.replace('images', 'labels')
val_images_dir = config['val']
val_labels_dir = val_images_dir.replace('images', 'labels')
test_images_dir = config['test']
test_labels_dir = test_images_dir.replace('images', 'labels')

class data(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_paths = [os.path.join(self.image_dir, img) for img in os.listdir(self.image_dir) if img.endswith('.jpg')]
        
        #reads labels and bounding box coords
        def load_yolo_labels(self, label_path):
            boxes = []
            labels = []

            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    class_index = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])

                    # Convert from YOLO format (x_center, y_center, width, height) to (x_min, y_min, x_max, y_max)
                    x_min = x_center - width / 2
                    y_min = y_center - height / 2
                    x_max = x_center + width / 2
                    y_max = y_center + height / 2

                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(class_index)

            return boxes, labels
        
        
        def __getitem__(self,idx):
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert("RGB")
            
            # apply necessary transformations
            if self.transform:
                image = self.transform(image)
            
            label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
            boxes, labels = self.load_yolo_labels(label_path)
            
            # Convert bounding boxes to tensor 
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)

            return image, {"boxes": boxes, "labels": labels}

# boilerplate code from ChatGPT
def train(model, num_epochs=10, learning_rate=0.001):
    # STEP 1: load the data into a Dataset object, establish loss functions and optimizer
    transform = transforms.Compose([
        transforms.ToTensor(), #converts image to tensor
        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)) #normalization adjust RGB mean, stdev here
    ])
    
    class_loss = nn.CrossEntropyLoss()
    box_loss = nn.SmoothL1Loss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # STEP 2: split the Datasets using the DataLoader
    train_dataset = data(train_images_dir, train_labels_dir, transform=transform)
    val_dataset = data(val_images_dir, val_labels_dir, transform=transform)

    training_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    validation_loader = DataLoader(val_dataset, batch_size = 4, shuffle=False)
    
   # STEP 3: begin training over number of epochs
        # STEP 3a: 
    model.train()
    for epoch in range(num_epochs):
        curr_loss = 0.0
        for images, labels in training_loader:
            class_labels = labels['labels']
            bbox_labels = labels['boxes']
            optimizer.zero_grad() # resets gradients of model param
            
            # Forward pass
            outputs = model(images)

            # Calculate losses
            class_pred, bbox_pred = outputs
            curr_class_loss += class_loss(class_pred, class_labels).item()
            curr_box_loss += box_loss(bbox_pred, bbox_labels).item()
            loss = curr_class_loss + curr_box_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            curr_loss += loss.item() #update loss
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {curr_loss/len(training_loader)}")
    
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # Disable gradient computation for validation
        for images, labels in validation_loader:
            class_labels = labels['labels']
            bbox_labels = labels['boxes']

            # Forward pass
            class_pred, bbox_pred = model(images)

            # Calculate losses
            class_loss_value = class_loss(class_pred, class_labels)
            box_loss_value = box_loss(bbox_pred, bbox_labels)
            loss = class_loss_value + box_loss_value

            val_loss += loss.item()