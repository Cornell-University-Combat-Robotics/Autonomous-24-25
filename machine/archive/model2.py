import torch
import torch.nn as nn

# boilerplate code (class header, init, self.___), from: 
# https://www.digitalocean.com/community/tutorials/writing-cnns-from-scratch-in-pytorch
class ConvNeuralNet(nn.Module):
    # initializing the layer architecture of the model; however, does not define the final ordering
    # assume a 600x600x3 image
    def __init__(self, num_classes=2, num_bots=3):
        print("initiating model")
        super(ConvNeuralNet, self).__init__()
        self.num_classes = num_classes
        self.num_bots = num_bots
        # print("finished super")
        self.conv_layer = nn.Conv2d(3, 10, 10) # output: 591 x 591 x 10 image
        # print("finished conv")
        self.relu = nn.ReLU() # output: 591 x 591 x 10 image
        # print("finsihed relu")
        feats_cut = 8
        self.max_pool = nn.MaxPool2d(feats_cut, feats_cut) # kernel for max pooling, changeable; kernel size 2x2, stride 2; effectively halves the dimensions
        # print("finished max pool")
        feats = 591//feats_cut
        self.lin1 = nn.Linear(feats * feats * 10, 5000) # in_channel: 591/4 (floored to 147) x 147 x 10
        # print("finsihed lin1")
        self.lin2 = nn.Linear(5000, 128) # sample and feature size, changeable
        # print("finished lin2")

        self.class_output = nn.Linear(128, num_classes * num_bots) # predicting on each bot with a number of classes
        self.box_output = nn.Linear(128, num_bots * 4) # for the four coordinates per bot; from chatgpt
        print("finished init model")
    # each step "forward" trains the parameters of the models, in the order specified by the layer architecture
    def forward(self, x):
        print("starting forward step")
        # convolutional layers
        out = self.conv_layer(x)
        out = self.relu(out)
        out = self.max_pool(out)

        # fully connected layers
        out = out.reshape(out.size(0), -1)
        out = self.lin1(out)
        out = self.lin2(out)

        # outputs
        class_scores = self.class_output(out)  # Shape: (batch_size, num_classes * num_bots)
        bounding_boxes = self.box_output(out)  # Shape: (batch_size, num_bots * 4)

        # Reshape outputs
        batch_size = x.size(0)
        num_bots = self.num_bots  # Make sure num_bots is an attribute of your model
        num_classes = self.num_classes  # Similarly, make num_classes an attribute

        class_scores = class_scores.view(batch_size, num_bots, num_classes)
        bounding_boxes = bounding_boxes.view(batch_size, num_bots, 4)

        return class_scores, bounding_boxes
