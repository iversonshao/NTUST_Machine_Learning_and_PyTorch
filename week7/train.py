import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from model import RNN
import numpy as np
from dataset import timeseries
import math
import matplotlib.pyplot as plt

lr = 1e-3
epochs = 101

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model = RNN().to(device)
loss_f = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr = lr)

a = np.arange(1, 721, 1)
b = np.sin(a * np.pi/180) + np.random.randn(720) * 0.05
b = b.reshape(-1, 1) 
X = []
Y = []
for i in range(710):
    input = []
    for j in range(i, i+10):
        input.append(b[j])
    X.append(input) 
    Y.append(b[j+1]) 

X = np.array(X)
Y = np.array(Y)

train_x = X[:360]
train_y = Y[:360]
test_x = X[360:]
test_y = Y[360:]


train_dataset = timeseries(train_x, train_y)
test_dataset = timeseries(test_x, test_y)
trainloader = DataLoader(train_dataset, batch_size = 32, shuffle = False, drop_last = True)
testloader = DataLoader(test_dataset, batch_size = 32, shuffle = False, drop_last = True)
#def train():
losses = []
for i in range(epochs):
    for _,(data,target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        
        opt.zero_grad()
        y_pred = model(data)
        loss = loss_f(y_pred, target)
        loss.backward()
        opt.step()    
    losses.append(loss.item())
    if i % 10 == 0:
        print(i, "th epoch : ",loss.item())
        torch.save(model.state_dict(), '{}.pt'.format(i))

#def test():
test_pred = model(test_dataset[:][0].view(-1, 10, 1).to(device)).view(-1)
pred = test_pred.cpu().detach().numpy() #order of cpu and detach is not important

plt.figure(1)
plt.plot(test_dataset[:][1], label = 'org')
plt.plot(pred, label = 'pred')
plt.legend()

plt.figure(2)
plt.plot(losses)
plt.show()