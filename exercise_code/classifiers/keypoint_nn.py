import torch
import torch.nn as nn
import torch.nn.functional as F


class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()

        #######################################################################
        # DONE: Define all the layers of this CNN, the only requirements are: #
        # 1. This network takes in a square (same width and height),          #
        #    grayscale image as input.                                        #
        # 2. It ends with a linear layer that represents the keypoints.       #
        # It's suggested that you make this last layer output 30 values, 2    #
        # for each of the 15 keypoint (x, y) pairs                            #
        #                                                                     #
        # Note that among the layers to add, consider including:              #
        # maxpooling layers, multiple conv layers, fully-connected layers,    #
        # and other layers (such as dropout or  batch normalization) to avoid #
        # overfitting.                                                        #
        #######################################################################
        # NOTE:
        # - For conv layers other parameters are as follows: stride=1,
        # padding=0, dilation=1, bias=True, which are by default.
        # - For pool layers using stride=kernel size (2), which is by default.
        # - For fc layers using bias=True, which is by default.
        #######################################################################
      
        conv1 = nn.Conv2d(1, 32, kernel_size=4)    # 1x96x96 -> 32x93x93
        active1 = nn.ELU()
        pool1 = nn.MaxPool2d(2)                    # 32x93x93 -> 32x46x46
        batch1 = nn.BatchNorm2d(32)
        drop1 = nn.Dropout(0.1)
        
        conv2 = nn.Conv2d(32, 64, kernel_size=3)   # 32x46x46 -> 64x44x44
        active2 = nn.ELU()
        pool2 = nn.MaxPool2d(2)                    # 64x44x44 -> 64x22x22
        batch2 = nn.BatchNorm2d(64)
        drop2 = nn.Dropout(0.2)
        
        conv3 = nn.Conv2d(64, 128, kernel_size=2)  # 64x22x22 -> 128x21x21
        active3 = nn.ELU()
        pool3 = nn.MaxPool2d(2)                    # 128x21x21 -> 128x10x10
        batch3 = nn.BatchNorm2d(128)
        drop3 = nn.Dropout(0.3)
        
        conv4 = nn.Conv2d(128, 256, kernel_size=1) # 128x10x10 -> 256x10x10
        active4 = nn.ELU()
        pool4 = nn.MaxPool2d(2)                    # 256x10x10 -> 256x5x5
        batch4 = nn.BatchNorm2d(256)
        drop4 = nn.Dropout(0.4)

        flat = nn.Flatten()                        # 256x5x5 -> 6400

        fc1 = nn.Linear(6400, 1000)                # 6400 -> 1000
        active5 = nn.ELU()
        drop5 = nn.Dropout(0.5)

        fc2 = nn.Linear(1000, 1000)                # 1000 -> 1000
        active6 = nn.ReLU()
        drop6 = nn.Dropout(0.6)

        fc3 = nn.Linear(1000, 30)                  # 1000 -> 30

        self.layers = nn.Sequential(
            conv1, active1, pool1, batch1, drop1,
            conv2, active2, pool2, batch2, drop2,
            conv3, active3, pool3, batch3, drop3,
            conv4, active4, pool4, batch4, drop4,
            flat,
            fc1, active5, drop5,
            fc2, active6, drop6,
            fc3
        )
        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################

    def forward(self, x):
        #######################################################################
        # DONE: Define the feedforward behavior of this model                 #
        # x is the input image and, as an example, here you may choose to     #
        # include a pool/conv step:                                           #
        # x = self.pool(F.relu(self.conv1(x)))                                #
        # a modified x, having gone through all the layers of your model,     #
        # should be returned                                                  #
        #######################################################################

        x = self.layers(x)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
