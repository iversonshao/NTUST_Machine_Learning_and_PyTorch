import torch
import matplotlib.pyplot as plt
from model import regression
from torch import nn, optim

#set data
x = torch.randn([100, 1])
w = torch.tensor([10.])
b = torch.tensor([3.])
y = w * x + b + torch.randn(x.shape) * 2.5
"""
print(y)
print(x)

plt.figure(1)
plt.plot(x, y, 'ro')
plt.show()
"""


device = torch.device('cuda')
model = regression(1,1).to(device)
cirterion = nn.MSELoss() #Set Loss function
opt = optim.SGD(model.parameters(), lr = 0.1) #lr->步伐大小

epoch = 10
losses = []

for i in range(epoch):
    x, y = x.to(device), y.to(device)
    model.train()

    opt.zero_grad()

    pred = model(x)
    loss = cirterion(y, pred)
    
    #loss之backward
    loss.backward()
    opt.step()

    losses.append(loss)
    #3.7version looses.append(loss.item()) 要加.item 否則畫不出來
    print('epoch:{}, loss:{}'.format(i,loss))


plt.figure(2)
plt.plot(losses, label = 'loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss')
plt.legend()
plt.savefig("loss.png")
plt.show()
