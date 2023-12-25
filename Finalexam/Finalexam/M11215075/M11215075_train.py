import torch 
from torch import nn, optim 
from torch.utils.data import DataLoader, random_split  
from model import LSTM 
import csv 
import pandas as pd 
import numpy as np 
import math 
import matplotlib.pyplot as plt
from dataset import wind

df = pd.read_csv("M11215075/data/wind_dataset.csv", header = 0)
dataset = wind(df)

#split train ana val
train_len = int(len(dataset) * 0.7)
val_len = len(dataset) - train_len
generator = torch.Generator().manual_seed(0)
train_data, val_data = random_split(dataset, [train_len, val_len], generator)

train_loader = DataLoader(train_data, batch_size = 32, shuffle = False, drop_last = True)
val_loader = DataLoader(val_data, batch_size = 32, shuffle = False, drop_last = True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM()
model = model.to(device)

loss_f = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr = 1e-3)

def train():
    model.train()
    train_loss = 0

    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        opt.zero_grad()
        pred = model(data)

        loss = loss_f(pred, target)

        loss.backward()
        opt.step()

        train_loss += loss.item()

    print("train_loss:{:.6f}".format(train_loss))
    return  train_loss 
def test():
    model.eval()
    val_loss = 0
    for idx, (data, target) in enumerate(val_loader):

        data, target = data.to(device), target.to(device)
        pred = model(data)
        pred = pred.view(-1)
        loss = loss_f(pred, target)
        val_loss += loss.item()
    print("val_loss:{:.6f}".format(val_loss))
    return val_loss

if __name__ == "__main__":
    train_losses = []
    val_losses = []

    for i in range(10):
        train_loss = train()
        val_loss = test()
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    torch.save(model.state_dict(), "M11215075_model.pt")

    plt.figure(1)
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.show()