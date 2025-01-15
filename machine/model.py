import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNeuralNet(nn.Module):
    def __init__(self, num_classes=2, num_bots=3):
        super(ConvNeuralNet, self).__init__()
        self.num_classes = num_classes
        self.num_bots = num_bots
        self.dropout = nn.Dropout(p=0.3)

        # Define convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.batchnorm1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.relu2 = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 32, 3, 1)
        self.relu3 = nn.ReLU()
        self.batchnorm3 = nn.BatchNorm2d(32)
        
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batchnorm4 = nn.BatchNorm2d(32)
        
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        self.batchnorm5 = nn.BatchNorm2d(32)
        
        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU()
        self.maxpool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batchnorm6 = nn.BatchNorm2d(32)
        
        # Define output layers
        self.prob_output = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.box_output = nn.Conv2d(32, 4, kernel_size=3, padding=1)
        self.class_output = nn.Conv2d(32, num_classes, kernel_size=3, padding=1)

        # Add adaptive pooling layer to ensure fixed output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, labels=None):
        if isinstance(labels, list):
            labels = labels[0]
        # Forward pass through convolutional layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.batchnorm1(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.batchnorm2(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.batchnorm3(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
        x = self.batchnorm4(x)
        x = self.dropout(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.batchnorm5(x)
        x = self.dropout(x)
        
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.maxpool6(x)
        x = self.batchnorm6(x)
        x = self.dropout(x)

       # Probability output for confidence scores
        prob_scores = self.prob_output(x)  # Shape: [batch_size, 1, height, width]
        prob_scores = prob_scores.view(-1)  # Flatten to [batch_size * height * width]

        # Get bounding boxes and class scores and flatten
        bounding_boxes = self.box_output(x)  # Shape: [batch_size, 4, height, width]
        bounding_boxes = bounding_boxes.permute(0, 2, 3, 1).contiguous().view(-1, 4)  # Shape: [batch_size * height * width, 4]

        class_scores = self.class_output(x)  # Shape: [batch_size, num_classes, height, width]
        class_scores = class_scores.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)  # Shape: [batch_size * height * width, num_classes]
                    
        # Select top-3 boxes based on confidence scores
        top_k = 3
        top_k_indices = torch.topk(prob_scores, top_k).indices  # Get indices of top-K confidence scores
        bounding_boxes = bounding_boxes[top_k_indices]  # Select the top-K bounding boxes
        bounding_boxes *= 600  # scale box coords
        class_scores = class_scores[top_k_indices]      # Select corresponding class scores
        
        if labels is not None:
            labels_flat = labels['labels'].view(-1)
            # print("Shape of labels_flat:", labels_flat.shape)
            
            # Calculate cross-entropy loss
            class_loss = F.cross_entropy(class_scores, labels_flat)

            # Ensure labels['boxes'] has the same shape as bounding_boxes for loss calculation
            labels_boxes = labels['boxes'].view(-1, 4)  # Flatten to [num_boxes, 4]
            
            # Print shapes for debugging
            # print("Shape of bounding_boxes (selected):", bounding_boxes.shape)
            # print("Shape of labels['boxes']:", labels_boxes.shape)

            # Calculate bounding box regression loss
            box_loss = F.smooth_l1_loss(bounding_boxes, labels_boxes, reduction='mean')

            # Total loss
            box_loss /= 600
            class_loss *= 2
            total_loss = class_loss + box_loss
            
            return {
                "loss_classifier": class_loss,
                "loss_box_reg": box_loss,
                "total_loss": total_loss
            }
        
        return class_scores, bounding_boxes, prob_scores