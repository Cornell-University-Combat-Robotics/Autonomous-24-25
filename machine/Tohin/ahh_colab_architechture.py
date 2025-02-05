import torch
import torch.nn as nn
import torch.nn.functional as F

class TohinNeuralNet(nn.Module):
    def __init__(self, num_classes=2, num_bots=3):
        super(TohinNeuralNet, self).__init__()
        self.num_classes = num_classes
        self.num_objects = num_bots

        self.conv1 = nn.Conv2d(3,32,3,1)
        self.relu1 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(32)

        self.maxpool1 = nn.MaxPool2d(3,2)

        self.conv2 = nn.Conv2d(32,32,3,1)
        self.relu2 = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm2d(32)

        self.maxpool2 = nn.MaxPool2d(3,2)

        self.conv3 = nn.Conv2d(32,32,3,1)
        self.relu3 = nn.ReLU()
        self.batchnorm3 = nn.BatchNorm2d(32)

        self.maxpool3 = nn.MaxPool2d(3,2)

        self.conv4 = nn.Conv2d(32,64,3,1)
        self.relu4 = nn.ReLU()
        self.batchnorm4 = nn.BatchNorm2d(64)

        self.maxpool4 = nn.MaxPool2d(3,2)

        self.conv5 = nn.Conv2d(64,64,3,1)
        self.relu5 = nn.ReLU()
        self.batchnorm5 = nn.BatchNorm2d(64)

        self.adaptive_pool = nn.AdaptiveMaxPool2d((4,4))

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.relu_fc = nn.ReLU()

        self.bbox_output = nn.Linear(512, self.num_objects * 4)
        self.class_output = nn.Linear(512, self.num_objects * self.num_classes)

    def forward(self, x, labels=None):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.batchnorm1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.batchnorm2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.batchnorm3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
        x = self.batchnorm4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.batchnorm5(x)

        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu_fc(x)

        bounding_boxes = self.bbox_output(x)
        bbox_pred = bounding_boxes.view(-1, self.num_objects, 4)
        class_scores = self.class_output(x)
        class_pred = class_scores.view(-1, self.num_objects, self.num_classes)

        if labels is not None:
            # ... rest of your training logic ...
            pass
        
        class_probs = F.softmax(class_pred, dim=-1)
        return class_probs, bbox_pred