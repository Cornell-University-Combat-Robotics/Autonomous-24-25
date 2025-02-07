import torch
import torch.nn as nn
import torch.nn.functional as F

class TohinNeuralNet(nn.Module):
    def __init__(self,num_classes=2,num_bots=3):
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

        # Posibly adaptive
        self.adaptive_pool = nn.AdaptiveMaxPool2d((4,4))

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.relu_fc = nn.ReLU()

        self.bbox_output = nn.Linear(512, self.num_objects * 4)  # Predict bounding boxes
        self.class_output = nn.Linear(512, self.num_objects * self.num_classes)

    def forward(self,x, labels=None):
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

        # Flatten the output
        x = self.flatten(x)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu_fc(x)

        bounding_boxes = self.bbox_output(x)
        bbox_pred = bounding_boxes.view(-1, self.num_objects, 4)
        bbox_pred = torch.sigmoid(bbox_pred)  # Normalize bbox predictions to [0,1]

        class_scores = self.class_output(x)
        class_pred = class_scores.view(-1, self.num_objects, self.num_classes)
        class_probs = F.softmax(class_pred, dim=-1)  # Convert to probabilities

        # class_scores = class_scores.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)  # Shape: [batch_size * height * width, num_classes]

        if labels is not None:
            labels_classes = labels['labels']  # [batch_size, num_objects]
            labels_boxes = labels['boxes']     # [batch_size, num_objects, 4]

            # Flatten predictions and labels
            pred_classes = class_pred.view(-1, self.num_classes)   # [batch_size * num_objects, num_classes]
            target_classes = labels_classes.view(-1)               # [batch_size * num_objects]

            pred_boxes = bbox_pred.view(-1, 4)                     # [batch_size * num_objects, 4]
            target_boxes = labels_boxes.view(-1, 4)                # [batch_size * num_objects, 4]

            # Masks for housebot and robots
            housebot_mask = target_classes == 0  # Assuming 0 is the class index for housebot
            robot_mask = target_classes == 1     # Assuming 1 is the class index for robots

            # Classification Losses
            housebot_class_loss = F.cross_entropy(
                pred_classes[housebot_mask],
                target_classes[housebot_mask]
            ) if housebot_mask.sum() > 0 else torch.tensor(0.0, device=x.device)

            robot_class_loss = F.cross_entropy(
                pred_classes[robot_mask],
                target_classes[robot_mask]
            ) if robot_mask.sum() > 0 else torch.tensor(0.0, device=x.device)

            # Regression Losses
            housebot_bbox_loss = F.smooth_l1_loss(
                pred_boxes[housebot_mask],
                target_boxes[housebot_mask]
            ) if housebot_mask.sum() > 0 else torch.tensor(0.0, device=x.device)

            robot_bbox_loss = F.smooth_l1_loss(
                pred_boxes[robot_mask],
                target_boxes[robot_mask]
            ) if robot_mask.sum() > 0 else torch.tensor(0.0, device=x.device)

            # Apply loss weights
            housebot_loss_weight = 0.5  # Adjust as needed
            robot_loss_weight = 1.0     # Adjust as needed

            housebot_total_loss = housebot_loss_weight * (housebot_class_loss + housebot_bbox_loss)
            robot_total_loss = robot_loss_weight * (robot_class_loss + robot_bbox_loss)
            total_loss = housebot_total_loss + robot_total_loss

            return {
                "loss_classifier_housebot": housebot_class_loss,
                "loss_box_reg_housebot": housebot_bbox_loss,
                "loss_classifier_robot": robot_class_loss,
                "loss_box_reg_robot": robot_bbox_loss,
                "total_loss": total_loss
            }
        class_probs = F.softmax(class_pred, dim=-1)
        return class_probs, bbox_pred