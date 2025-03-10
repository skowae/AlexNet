# Autoencoder.py 
# Andrew Skow
# Deep Learning For Computer Vision EN.525.733
# March 9, 2025

import torch
import torch.nn as nn

# AlexNet
# This class defines the architecture of my AlexNet like model architecture.
class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()

        # Define the first convolutional layer.  3 input channels for RGB.
        # I use batch normalization to normalize after ReLU activation
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=3, 
                      stride=4, padding=0),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        # Define the second convolutional layer.  Stride will be 1.  So will 
        # all subsequent conv layers.  Padding is 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3,
                      stride=1, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # Define the third convolutional layer.  There is no max pooling in 
        # this layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3, 
                      stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )

        # Define the fourth convolutional layer.  Similar to layer 3
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, 
                      stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )


        # Define the first fully connected layer.  I will use a 0.5 drop out.
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=(14*14*192), out_features=512),
            nn.ReLU(inplace=True)
        )

        # Define the second fully conected layer
        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(inplace=True)
        )

        # Define the final classification output
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=256, out_features=num_classes),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        '''
        The forward function runs inference on the model.  Given and input the 
        forward function passes the data through the model and returns the 
        output.
        
        Param (x): The input array.  In our case a 32x32 image

        Return The output of the model
        '''
        # Flatten the image into a vector
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out