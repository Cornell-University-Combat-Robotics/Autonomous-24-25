import datetime
import os
from dotenv import load_dotenv
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import yaml
import model
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import default_collate
from roboflow import Roboflow

# Load environment variables
load_dotenv()

# Roboflow dataset details
DATASET_DETAILS = {
    "workspace": "crc-autonomous",
    "project": "nhrl-robots",
    "version": 3
}

def roboflow_download(dataset_name, save_dir="data2"):
    """Downloads the specified dataset from Roboflow using the API."""
    roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")
    rf = Roboflow(api_key=roboflow_api_key)
    project = rf.workspace(DATASET_DETAILS['workspace']).project(DATASET_DETAILS['project'])
    version = project.version(DATASET_DETAILS['version'])
    version.download("yolov8", location=os.path.join(save_dir, dataset_name))

class Data(Dataset):
    """Custom PyTorch Dataset class that downloads data from Roboflow if not available locally."""
    def __init__(self, dataset_name="NHRL", transform=None):
        self.dataset_name = dataset_name
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Directories
        data_dir = os.path.join("data", dataset_name)
        self.image_dir = os.path.join(data_dir, "train/images")
        self.label_dir = os.path.join(data_dir, "train/labels")

        # Check if data exists; if not, download from Roboflow
        if not os.path.exists(self.image_dir) or not os.listdir(self.image_dir):
            print(f"{dataset_name} data not found locally. Downloading from Roboflow.")
            roboflow_download(dataset_name)

        # Populate image paths
        self.image_paths = [os.path.join(self.image_dir, img) for img in os.listdir(self.image_dir) if img.endswith('.jpg')]
        print(f"Found {len(self.image_paths)} images in {self.image_dir}")

    def load_yolo_labels(self, label_path):
        # (Same as your current implementation)
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

    def __getitem__(self, idx):
        # (Same as your current implementation)
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
        boxes, labels = self.load_yolo_labels(label_path)
        
        if not boxes:
            print(f"No boxes found in file: {img_path}")
            return None
                    
        if len(labels) != 3:
            print(f"Skipping image {img_path} due to incorrect number of labels: {len(labels)}")
            return None
            
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        if len(labels) != len(boxes):
            print(f"Mismatch in number of boxes and labels in file {img_path}: {len(boxes)} boxes, {len(labels)} labels")
            return None
    
        return image, {"boxes": boxes, "labels": labels}
    
    def __len__(self):
        return len(self.image_paths)

def train(model, num_epochs=10, learning_rate=0.0001):
    print("begin training")
    writer = SummaryWriter(log_dir=f"runs/experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Datasets and DataLoaders
    train_dataset = Data(dataset_name="NHRL", transform=transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
    val_dataset = Data(dataset_name="NHRL", transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
    def collate_fn(batch):
        # Filter out None items
        batch = [item for item in batch if item is not None]
        # If batch is empty, return None to indicate it should be skipped
        return default_collate(batch) if len(batch) > 0 else None

    training_loader = DataLoader(train_dataset, batch_size = 4, shuffle=True, collate_fn=collate_fn)
    validation_loader = DataLoader(val_dataset, batch_size = 4, shuffle=False, collate_fn=collate_fn)

    def loader_loss(images, labels):
        """
        Calculates total loss (classification and regression) for given images and labels
        """
        # print(f"shape: {labels['boxes'].shape} ")
        if labels is None or 'boxes' not in labels or labels['boxes'].dim() != 3 or labels['boxes'].shape[-1] != 4:
            # print(labels['boxes'])

            print("Skipping batch due to missing or malformed labels")
            return None
        #squeeze for batch size 1 only
        # labels['boxes'] = labels['boxes'].squeeze(0)  # Shape should be [num_boxes, 4]
        # labels['labels'] = labels['labels'].squeeze(0)
        labels['boxes'] *= 600   # Scale 
        # print(f"Images shape: {images.shape}")  # Debug print
        # print(f"Labels structure (before passing to model): {labels}")
        # outputs = model(images,[labels])
        # outputs = model(images, [labels[i] for i in range(len(labels['boxes']))])
        labels = [{'boxes': labels['boxes'][i], 'labels': labels['labels'][i]} for i in range(len(labels['boxes']))]
        outputs = model(images, labels)

        
        if isinstance(outputs, dict):
            total_loss = outputs['total_loss'].mean()
            class_loss = outputs['loss_classifier'].mean()
            box_loss = outputs['loss_box_reg'].mean()
            
            # Calculate probability loss with Binary Cross Entropy on class_probs
            return total_loss, class_loss, box_loss  # Return all three losses for logging purposes
        else:
            raise TypeError("Expected a dictionary of losses, but got:", type(outputs))
            
    # print("model is in training")
    # print(f"Batch size: {training_loader.batch_size}")
    # print(f"Collate function: {training_loader.collate_fn}")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(training_loader):
            if batch is None: continue
            images, labels = batch
            print(f"Step {i + 1}/{len(training_loader)}")
            optimizer.zero_grad()
            loss = loader_loss(images, labels)
            if loss is None:
                continue
            total_loss, class_loss, box_loss = loss
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
            # Log loss to TensorBoard
            writer.add_scalar('Training Total Loss', total_loss.item(), epoch * len(training_loader) + i)
            writer.add_scalar('Training Classification Loss', class_loss.item(), epoch * len(training_loader) + i)
            writer.add_scalar('Training Box Regression Loss', box_loss.item(), epoch * len(training_loader) + i)        
        avg_loss = running_loss / len(training_loader)
        writer.add_scalar('Average Loss per Epoch', avg_loss, epoch)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(training_loader)}")
    
        model.eval() 
        print("model is in validation")
        val_loss = 0.0
        with torch.no_grad(): 
            for i, batch in enumerate(validation_loader):
                if batch is None: continue
                images, labels = batch
                loss = loader_loss(images, labels)
                if loss is None: continue
                valid_loss, _, _ = loss
                val_loss += valid_loss.item()
        avg_val_loss = val_loss / len(validation_loader)
        writer.add_scalar('Validation Loss', avg_val_loss, epoch)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss}")
        
    print("end training")

def main():
    newModel = model.ConvNeuralNet()
    # newModel = fasterrcnn_resnet50_fpn(pretrained=True)
    print("made model")
    # roboflow_download()
    train(newModel, num_epochs=10)
    print("finished training model")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(newModel, f"./models/model_{timestamp}.pth")
    print("saved new model")

if __name__ == "__main__":
    main()