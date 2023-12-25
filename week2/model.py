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
#super()用來調用父類的方法，__init__()是類的構造方法
#super().__init__() 就是調用父類的init方法，同樣可使用super()去調用父類的其他方法
    def forward(self, x):
        x = self.linear(x)

        return x

       
    

if __name__ == '__main__':
    device = torch.device('cuda')
    model = regression(1, 1).to(device)
    summary(model,(1, 1))


    