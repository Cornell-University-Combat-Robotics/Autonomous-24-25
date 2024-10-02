import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# boilerplate code from ChatGPT
def train(model, num_epochs=10, learning_rate=0.001):
    # STEP 1: load the data into a Dataset object, establish loss functions and optimizer
    # for some reason we need to define transformations we apply?
    transforms = transforms.Compose(
        # turns the image into PyTorch tensors
        transforms.ToTensor()
    )
    class_loss = nn.CrossEntropyLoss()
    box_loss = nn.SmoothL1Loss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # STEP 2: split the Datasets using the DataLoader

    # STEP 3: begin training over number of epochs
        # STEP 3a: 