import torch
from torch import nn
#import torch.nn.functional as F
from torchsummary import summary

class CNN(nn.Module):
    def __init__(self, out_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, out_dim)
        self.sigmoid = nn.Sigmoid()
        #self.dropout = nn.Dropout(0.25)
    def forward(self, x):
        #x = self.maxpool(F.relu(self.conv1(x)))
        #x = self.maxpool(F.relu(self.conv2(x)))
        #x = self.maxpool(F.relu(self.conv3(x)))
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(-1, 128 * 6 * 6)
        #x = self.dropout(x)
        #x = F.hardtanh(self.fc1(x))
        x = self.fc1(x)
        #x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x 

if __name__ == '__main__':
    device = torch.device('cuda')
    model = CNN(150).to(device)
    summary(model, (3, 64, 64))