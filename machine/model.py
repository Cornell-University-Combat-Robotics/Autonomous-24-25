import torch
import torch.nn as nn

# boilerplate code (class header, init, self.___), from: 
# https://www.digitalocean.com/community/tutorials/writing-cnns-from-scratch-in-pytorch
class ConvNeuralNet(nn.Module):
    # initializing the layer architecture of the model; however, does not define the final ordering
    # assume a 600x600x3 image
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()
        self.conv_layer = nn.Conv2D(3, 10, 10) # output: 576x576x5 image
        self.max_pool = nn.MaxPool2d(10) # kernel for max pooling, changeable
        self.lin = nn.Linear(1600,128) # sample and feature size, changeable
        self.relu = nn.ReLU() # output: 576x576x5 image
        self.lin = nn.Linear(128,num_classes)
        
    # each step "forward" trains the parameters of the models, in the order specified by the layer architecture
    def forward(self, x):
        out = self.conv_layer(x)
        out = self.max_pool(out)

        out = out.reshape(out.size(0), -1)
        
        out = self.lin(out)
        out = self.relu(out)
        out = self.lin(out)

        return out
