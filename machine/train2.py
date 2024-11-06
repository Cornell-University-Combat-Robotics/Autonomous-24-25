import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import yaml
import model2
from torch.utils.tensorboard import SummaryWriter


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        
        if not boxes:
            print(f"No boxes found in file: {img_path}")
        else:
            # Check if each box has exactly 4 coordinates
            for i, box in enumerate(boxes):
                if len(box) != 4:
                    print(f"Incorrect box dimensions in file {img_path} at box index {i}: {box}")
            
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        # filename = os.path.basename(img_path)
        if len(labels) != len(boxes):
            print(f"Mismatch in number of boxes and labels in file {img_path}: {len(boxes)} boxes, {len(labels)} labels")
    
        return image, {"boxes": boxes, "labels": labels}
    
    def __len__(self):
        return len(self.image_paths)

def train(model, num_epochs=10, learning_rate=0.001):
    def loader_loss(images, targets):
        """
        Calculates total loss (classification and regression) for given images and targets.
        """
        # Move images and targets to the appropriate device (CPU or GPU)
        images = images.to(device)
        target_labels = targets['labels'].to(device)
        target_boxes = targets['boxes'].to(device)

        # Forward pass: get predictions from the model
        class_scores, bounding_boxes = model(images)

        # Reshape outputs and targets if necessary
        num_samples = class_scores.size(0)  # Batch size
        num_bots = target_labels.size(1)    # Number of bots per image

        # Reshape class_scores and target_labels to (batch_size * num_bots, num_classes)
        class_scores = class_scores.view(num_samples * num_bots, -1)
        target_labels = target_labels.view(-1)

        # Reshape bounding_boxes and target_boxes to (batch_size * num_bots, 4)
        bounding_boxes = bounding_boxes.view(num_samples * num_bots, -1)
        target_boxes = target_boxes.view(-1, 4)

        # Compute classification loss
        classification_loss = nn.CrossEntropyLoss()(class_scores, target_labels)

        # Compute regression loss
        regression_loss = nn.SmoothL1Loss()(bounding_boxes, target_boxes)

        # Total loss
        total_loss = classification_loss + regression_loss

        return total_loss

    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
    ])
    writer = SummaryWriter(log_dir='runs/experiment')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = Data(train_images_dir, None, transform=transform)
    val_dataset = Data(val_images_dir, None, transform=transform)

    training_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    validation_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(training_loader):
            optimizer.zero_grad()
            loss = loader_loss(images, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            writer.add_scalar('Training Loss', loss.item(), epoch * len(training_loader) + i)
        avg_loss = running_loss / len(training_loader)
        writer.add_scalar('Average Loss per Epoch', avg_loss, epoch)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in validation_loader:
                loss = loader_loss(images, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(validation_loader)
        writer.add_scalar('Validation Loss', avg_val_loss, epoch)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss}")
    writer.close()

def main():
    newModel = model2.ConvNeuralNet(num_classes=2, num_bots=3).to(device)
    train(newModel)
    torch.save(newModel.state_dict(), "./models/model.pth")
    print("Model saved.")

if __name__ == "__main__":
    main()