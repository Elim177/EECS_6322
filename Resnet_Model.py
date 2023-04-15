import torch
import torch.nn as nn
import torch.nn.functional as functional

# Define the BasicBlock module
# This is based on th torch vision repository

class BasicBlock(nn.Module):
    # Set the expansion factor for the residual block
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last_layer=False):
        super(BasicBlock, self).__init__()
        #this is the first convolution layer
        self.convolutional_layer_1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #batch norm for convolutional_layer_1
        self.batch_norm_1 = nn.BatchNorm2d(planes)
        #this is the second convolution layer
        self.convolutional_layer_2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        #batch norm for convolutional_layer_2
        self.batch_norm_2 = nn.BatchNorm2d(planes)
        #boolean to check if it is last layer
        self.is_last_layer = is_last_layer
        #this is the shortcut connection
        self.shortcut = nn.Sequential()
        #check the stride value and adjust the shortcut layer accordingly
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        #perform the convolution, batch normalization, and ReLU
        output = functional.relu(self.batch_norm_1(self.convolutional_layer_1(x)))
        #perform the second convolution and batch normalization
        output = self.batch_norm_2(self.convolutional_layer_2(output))
        #add the shortcut connection
        output = output + self.shortcut(x)
        #store the preactivation value
        preactivation = output
        #perform RELU activation on the outputput again
        output = functional.relu(output)
        #check if it is the last layer and return the out_put and pre-activation out_put
        if self.is_last_layer:
            return output, preactivation
        else:
            return output
        
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, grad_zeros=False):
        super(ResNet, self).__init__()
        #number of input planes
        self.in_planes = 64
        #initial convolutional layer
        self.convolutional_layer_1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm_1 = nn.BatchNorm2d(64)
        # these are the residual layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # global average pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # reset the residual layer weights to zero
        if grad_zeros:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.batch_norm_2.weight, 0)

    #  a function to make layers given block, plabes, number of blocs and stride
    def _make_layer(self, block, planes, num_blocks, stride):
        # instructions to make new stirdes and layers
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        #simple forward feed
        out_put_val = functional.relu(self.bn1(self.conv1(x)))
        out_put_val = self.layer1(out_put_val)
        out_put_val = self.layer2(out_put_val)
        out_put_val = self.layer3(out_put_val)
        out_put_val = self.layer4(out_put_val)
        out_put_val = self.avgpool(out_put_val)
        out_put_val = torch.flatten(out_put_val, 1)
        return out_put_val

#intialization of resnet18 which is made from the basic block and 4 layers 
#and has an output dimension of 512
def resnet18(**kwargs):
    return {'backbone': ResNet(BasicBlock, [2, 2, 2, 2], **kwargs), 
            'dim': 512}


