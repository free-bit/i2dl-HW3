"""SegmentationNN"""
import torch
import torch.nn as nn
from torchvision import models


class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        vgg16 = models.vgg16(pretrained=True)
        # print(vgg16.classifier)
        vgg16.classifier = nn.Sequential(
            # Input is 512x7x7, channel: 512->4096, spatial size: 7x7->1x1, yielding a vector of size 4096x1x1 (equivalent to output of FC1).
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(True),
            nn.Dropout(),
            # Input is 4096x1x1, channel: 4096->4096, spatial size: 1x1->1x1, yielding a vector of size 4096x1x1 (equivalent to output of FC2).
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(True),
            nn.Dropout(),
            # Input is 4096x1x1, channel: 4096->num_classes, spatial size: 1x1->1x1, yielding a vector of size num_classesx1x1 (equivalent to output of FC3).
            nn.Conv2d(4096, num_classes, 1),
            # TODO: Upsample parameters should be learned -> Use Conv2dTranspose

        )
        # print(vgg16.classifier)
        self.aug_vgg16 = vgg16 # nn.Module
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        output_size = x.size()[2:4] # TODO: comment this later
        features = self.aug_vgg16.features
        avgpool = self.aug_vgg16.avgpool
        classifier = self.aug_vgg16.classifier
        for layer in features:
            # print(layer)
            x = layer(x)
        
        # print(avgpool)
        x = avgpool(x)

        for layer in classifier:
            # print(layer)
            x = layer(x)

        # Upsample the output by a factor of 32 with bilinear interpolation (TODO: For testing purposes, comment later)
        layer = nn.Upsample(size=output_size, mode='bilinear')
        # print(layer)
        x = layer(x)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
