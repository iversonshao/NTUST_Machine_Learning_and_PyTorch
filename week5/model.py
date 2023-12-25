import torch
from torch import nn
from torchsummary import summary

class DNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(input_dim, 500)
        self.linear2 = nn.Linear(500, 250)
        self.linear3 = nn.Linear(250, 125)#ReLu也是活化函數也可以用
        self.linear4 = nn.Linear(125, output_dim) #共三層隱藏層
        self.sigmoid = nn.Sigmoid() #sigmoid是機率函數，把輸出轉為機率

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.sigmoid(x)

        return x
    
if __name__ == '__main__':
    device = torch.device('cuda')
    model = DNN(784, 10).to(device)

    summary(model, (1, 784))
