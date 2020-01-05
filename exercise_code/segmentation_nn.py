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
        # vgg16.features[0].padding = (100, 100)
        old_classifier = vgg16.classifier
        
        # Input is 512x7x7, channel: 512->4096, spatial size: 7x7->1x1, yielding a vector of size 4096x1x1 (equivalent to output of FC1).
        fcn1 = nn.Conv2d(512, 4096, kernel_size=7)
        # Copy old parameters from fc1, reuse them in fcn1
        w_fc1 = old_classifier[0].weight.data.view(4096, 512, 7, 7)
        b_fc1 = old_classifier[0].bias.data
        fcn1.weight.data.copy_(w_fc1)
        fcn1.bias.data.copy_(b_fc1)
        # ReLU + Dropout
        r1 = nn.ReLU(inplace=True)
        d1 = nn.Dropout()

        # Input is 4096x1x1, channel: 4096->4096, spatial size: 1x1->1x1, yielding a vector of size 4096x1x1 (equivalent to output of FC2).
        fcn2 = nn.Conv2d(4096, 4096, kernel_size=1)
        # Copy old parameters from fc2, reuse them in fcn2
        w_fc2 = old_classifier[3].weight.data.view(4096, 4096, 1, 1)
        b_fc2 = old_classifier[3].bias.data
        fcn2.weight.data.copy_(w_fc2)
        fcn2.bias.data.copy_(b_fc2)
        # ReLU + Dropout
        r2 = nn.ReLU(inplace=True)
        d2 = nn.Dropout()
        
        # Input is 4096x1x1, channel: 4096->num_classes, spatial size: 1x1->1x1, yielding a vector of size num_classesx1x1 (equivalent to output of FC3).
        score = nn.Conv2d(4096, num_classes, kernel_size=1)
        score.weight.data.zero_()
        score.bias.data.zero_()
        vgg16.classifier = nn.Sequential(fcn1, r1, d1, fcn2, r2, d2, score)
        self.aug_vgg16 = vgg16 # nn.Module
        
        # x Upsample parameters should be learned, therefore we need Conv2dTranspose.
        # print(vgg16.classifier)
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
        output_size = x.size()[2:4] # Get spatial dimensions (i.e HxW for the input images)
        
        # Sets of layers in the network
        features = self.aug_vgg16.features
        avgpool = self.aug_vgg16.avgpool
        classifier = self.aug_vgg16.classifier
        upsample = nn.Upsample(size=output_size, mode='bilinear')

        # for layer in features:
        #     x = layer(x)
            # print(x.size())
        
        # x = avgpool(x)
        # print(x.size())

        # for layer in classifier:
        #     x = layer(x)
            # print(x.size())

        # Upsample the output by a factor of 32 with bilinear interpolation (TODO: For testing purposes, comment later)
        #layer = nn.Upsample(size=output_size, mode='bilinear')
        x = features(x)
        x = avgpool(x)
        x = classifier(x)
        x = upsample(x)
        # Output shape is: B x (num_classes x H x W) -> num_classes-many heat maps per image
        # PyTorch is able to calculate the loss over different class predictions/heat maps with cross entropy loss automatically.
        # Therefore, a reduction over the dim-1 (i.e. num_classes) is not needed.
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
