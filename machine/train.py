import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import yaml
import model


# print("opening yaml")
with open('./data/yolo_data_v1/nhrl_bots.yaml','r') as file: #edit yaml path
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

def train(model, num_epochs=10, learning_rate=0.001):
    print("begin training")
    """
    Performs training and evaluation of the model
    """
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])
    
    class_loss = nn.CrossEntropyLoss()
    box_loss = nn.SmoothL1Loss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = Data(train_images_dir, train_labels_dir, transform=transform)
    val_dataset = Data(val_images_dir, val_labels_dir, transform=transform)

    training_loader = DataLoader(train_dataset, batch_size = 4, shuffle=True)
    validation_loader = DataLoader(val_dataset, batch_size = 4, shuffle=False)

    def loader_loss(images, labels):
        """
        Calculates total loss (classification and regression) for given images and labels
        """
        class_labels = labels['labels']
        # class_labels = torch.argmax(labels['labels'], dim=1)
        print(class_labels)
        bbox_labels = labels['boxes']
        
        class_pred, bbox_pred = model.forward(images)
        curr_class_loss = 0
        curr_box_loss = 0
        curr_class_loss += class_loss(class_pred, class_labels)
        curr_box_loss += box_loss(bbox_pred, bbox_labels)
        return curr_class_loss + curr_box_loss
    model.train()
    print("model is in training")
    print(f"Batch size: {training_loader.batch_size}")
    print(f"Collate function: {training_loader.collate_fn}")
    for epoch in range(num_epochs):
        curr_loss = 0.0
        for images, labels in training_loader:
            print(f"Image batch shape: {images.shape}")
            for key, value in labels.items():
                    print(f"Label {key} shape: {value.shape}")            
            optimizer.zero_grad()
            loss = loader_loss(images, labels)
            loss.backward()
            optimizer.step()
            curr_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {curr_loss/len(training_loader)}")
    
    model.eval() 
    print("model is in validation")
    val_loss = 0.0
    with torch.no_grad(): 
        for images, labels in validation_loader:
            loss = loader_loss(images, labels)
            val_loss += loss.item()
    print("end training")

def main():     
    newModel = model.ConvNeuralNet()
    print("made model")
    train(newModel)
    print("finished training model")
    torch.save(newModel, "./models")
    print("saved new model")

if __name__ == "__main__":
    main()