import datetime
import os
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
        
        print(f"Found {len(self.image_paths)} images in {self.image_dir}")

    
    
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
        # print(f"Processing image: {img_path}")
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
        # filename = os.path.basename(img_path)
        if len(labels) != len(boxes):
            print(f"Mismatch in number of boxes and labels in file {img_path}: {len(boxes)} boxes, {len(labels)} labels")
            return None
    
        return image, {"boxes": boxes, "labels": labels}
    
    def __len__(self):
        return len(self.image_paths)

def train(model, num_epochs=10, learning_rate=0.001):
    print("begin training")
    """
    Performs training and evaluation of the model
    """
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])
    log_dir = f"runs/experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    def collate_fn(batch):
        # Filter out None items
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None  # If the entire batch is empty, return None
        return torch.utils.data.dataloader.default_collate(batch)

    train_dataset = Data(train_images_dir, train_labels_dir, transform=transform)
    print(f"length: {len(train_dataset)}")
    val_dataset = Data(val_images_dir, val_labels_dir, transform=transform)

    training_loader = DataLoader(train_dataset, batch_size = 1, shuffle=True, collate_fn=collate_fn)
    validation_loader = DataLoader(val_dataset, batch_size = 1, shuffle=False)

    def loader_loss(images, labels):
        """
        Calculates total loss (classification and regression) for given images and labels
        """
        # print(f"shape: {labels['boxes'].shape} ")
        if labels is None or 'boxes' not in labels or labels['boxes'].shape != torch.Size([1,3,4]):
            print(labels['boxes'])

            print("Skipping batch due to missing or malformed labels")
            return None
        #squeeze for batch size 1 only
        labels['boxes'] = labels['boxes'].squeeze(0)  # Shape should be [num_boxes, 4]
        labels['labels'] = labels['labels'].squeeze(0)
        # print(f"Images shape: {images.shape}")  # Debug print
        # print(f"Labels structure (before passing to model): {labels}")
        outputs = model(images,[labels])
        
        if isinstance(outputs, dict):
            total_loss = outputs['total_loss']
            class_loss = outputs['loss_classifier']
            box_loss = outputs['loss_box_reg']
            
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
    
        # model.eval() 
        # print("model is in validation")
        # val_loss = 0.0
        # with torch.no_grad(): 
        #     for images, labels in validation_loader:
        #         loss = loader_loss(images, labels)
        #         val_loss += loss.item()
        # avg_val_loss = val_loss / len(validation_loader)
        # writer.add_scalar('Validation Loss', avg_val_loss, epoch)
        # print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss}")
        
    print("end training")

def main():
    newModel = model.ConvNeuralNet()
    # newModel = fasterrcnn_resnet50_fpn(pretrained=True)
    print("made model")
    train(newModel, num_epochs=10)
    print("finished training model")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(newModel, f"./models/model_{timestamp}.pth")
    print("saved new model")

if __name__ == "__main__":
    main()