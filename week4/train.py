import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from model import regression
from dataset import homeworkData

import csv
import pandas as pd
import numpy as np
import math

#for plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

batch_size = 1
lr = 1e-3
epochs = 140
device = torch.device('cuda')

df = pd.read_csv("week4/Data/1,01.csv", header = 0)
df = df.to_numpy(dtype = np.float32)
df = torch.from_numpy(df)
data = homeworkData(df)


#data split
train_len = int(len(data) // 3)
test_len = len(data) - train_len*2
train_data, val_data, test_data = random_split(data, [train_len, train_len, test_len])

train_data = data
train_loader = DataLoader(train_data, batch_size = batch_size)
val_loader = DataLoader(val_data, batch_size = batch_size)
#test_loader = DataLoader(test_data, batch_size = batch_size)

#Loss & Optimizer
model = regression(1, 1).to(device)
loss_f = nn.MSELoss()
opt = optim.Adagrad(model.parameters(), lr = lr)

def train(epoch):
    model.train()
    losses = 0

    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        opt.zero_grad()

        pred = model(data)
        loss = loss_f(target, pred)
        losses += loss
        loss.backward()
        opt.step()

        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), 'regression_{}.pt'.format(epoch))

    print("train_epoch:{}, train_loss:{:.6f}".format(epoch+1, losses/len(train_loader)))
    return losses/len(train_loader)

def val(epoch):
    model.eval()
    losses = 0
    
    for idx, (data, target) in enumerate(val_loader): 
        data, target = data.to(device), target.to(device)

        pred = model(data)
        loss = loss_f(pred, target)
        losses += loss
    
    print("val epoch:{}, loss:{:.2f}".format(epoch + 1, losses/len(val_loader)))
    
    return(losses/len(val_loader))

def test():
    model.eval()
    acc = 0
    total = 0
    with torch.no_grad(): 
        for data, target in test_data:
            data, target = data.to(device), target.to(device)
            pred = model(data).item()
            error = pred - target.item()
            acc += math.pow(error, 2)
            total += target.size(0)
        mse = math.sqrt(acc / total)
    
    return mse

if __name__ == '__main__':
    train_loss = []
    val_loss = []

    for i in range(epochs):
        train_loss.append(train(i))
        val_loss.append(val(i))

    #RMSE = test()
    #print("test RMSE:{:.2f}".format(RMSE))

    plt.figure(1)
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("loss")
    plt.legend()
    plt.show()

   
