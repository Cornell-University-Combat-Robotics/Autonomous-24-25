import torch
import torch.nn as nn

# boilerplate code (class header, init, self.___), from: 
# https://www.digitalocean.com/community/tutorials/writing-cnns-from-scratch-in-pytorch
class ConvNeuralNet(nn.Module):
    # initializing the layer architecture of the model; however, does not define the final ordering
    # assume a 600x600x3 image
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()
        self.conv_layer = nn.Conv2D(3, 10, 10) # output: 591x591x10 image
        self.relu = nn.ReLU() # output: 591x591x5 image
        self.max_pool = nn.MaxPool2d(2, 2) # kernel for max pooling, changeable
        self.lin1 = nn.Linear(870250, 87000) # in_channel: 591/2*591/2*10
        self.lin2 = nn.Linear(87000,128) # sample and feature size, changeable
        self.output = nn.Linear(128,num_classes + 4) # num classes plus 4 coordinates
        
    # each step "forward" trains the parameters of the models, in the order specified by the layer architecture
    def forward(self, x):
        out = self.conv_layer(x)
        out = self.relu(out)
        out = self.max_pool(out)

        out = out.reshape(out.size(0), -1)
        out = self.lin1(out)
        out = self.lin2(out)
        out = self.output(out)

        class_scores = out[:num_classes]
        bounding_boxes = out[-num_classes:]

        return class_scores, bounding_boxes
