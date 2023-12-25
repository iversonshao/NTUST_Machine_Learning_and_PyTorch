import torch
from torch import nn
from torchsummary import summary

class residualblock(nn.Module):
    def __init__(self, input_channel, output_channel, stride = 1): #stride = 1 -> intialize stride
        super().__init__()
        self.left = nn.Sequential(
                        nn.Conv2d(input_channel, output_channel, 3, stride = stride, padding = 1, bias = False),
                        nn.BatchNorm2d(output_channel),
                        nn.ReLU(inplace = True),
                        nn.Conv2d(output_channel, output_channel, 3, stride = 1, padding = 1, bias = False),
                        nn.BatchNorm2d(output_channel)
                        )
        self.shortcut = nn.Sequential()
        if stride != 1 or input_channel != output_channel:
            self.shortcut = nn.Sequential(
                            nn.Conv2d(input_channel, output_channel, 1, stride = stride, bias = False),
                            nn.BatchNorm2d(output_channel)
                            )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x) #determine x whether to  process
        out = self.relu(out)
        return out

#ResNet18 layer has 7*7*1, (3*3 residual block->4 layer) * 4, fully connected layer*1     
class ResNet(nn.Module): 
    def __init__(self, img_size, residualblock, num_classes):
        super(ResNet, self).__init__()
        self.inchannel = 64 #default channel

        #first convolution layer
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, 7, stride = 2, padding = 3, bias = False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        )
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        #4 block of residual layer
        self.layer1 = self.make_layer(residualblock, 64, 2, stride = 1)
        img_size //= 8 #224->28
        self.layer2 = self.make_layer(residualblock, 128, 2, stride = 2)
        img_size //= 2 #28->14
        self.layer3 = self.make_layer(residualblock, 256, 2, stride = 2)
        img_size //= 2 #14->7
        self.layer4 = self.make_layer(residualblock, 512, 2, stride = 2)
        #Global Average Pooling
        self.avgpool = nn.AvgPool2d(4)
        img_size //= 4 #7->1
        self.flatten = nn.Flatten()
        #fully connected layer
        self.fc = nn.Linear(512 * img_size * img_size, num_classes)

    def make_layer(self, residualblock, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        #print(strides)
        layers = []
        for stride in strides:
            layers.append(residualblock(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc(out)

        return out
    
if __name__ == '__main__':
    device = torch.device('cuda')
    model = ResNet(224, residualblock, 2).to(device)
    summary(model, (3, 224, 224))
    #print(model)