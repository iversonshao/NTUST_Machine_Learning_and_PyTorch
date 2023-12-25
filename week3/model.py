import torch
from torch import nn
from torchsummary import summary

#model最好自己寫一個
#regression -> 找一個函式


#線性regression

class regression(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(regression, self).__init__()
        self.linear = nn.Linear(input_dim, out_dim)

    def forward(self, x):
        x = self.linear(x)

        return x
    

if __name__ == '__main__':
    device = torch.device('cuda')
    model = regression(87, 1).to(device)
    summary(model,(1, 87))


    