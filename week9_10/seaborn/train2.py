import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from dataset import seaborndataset
from model import LSTM
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset("flights")
dataset = seaborndataset(df)

train_len = int(len(dataset) * 0.7)
test_len = len(dataset) - train_len
generator = torch.Generator().manual_seed(0)
train_data, test_data = random_split(dataset, [train_len, test_len], generator)

train_loader = DataLoader(train_data, batch_size = 16, shuffle = False, drop_last = True)
test_loader = DataLoader(test_data, batch_size = 16, shuffle = False, drop_last = True)
#print("data len:{}\ntrain len:{}\nvalid len:{}".format(len(dataset), len(train_data), len(valid_data)))

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
        # pred = pred.view(-1) #if self.normalized_dataset = self.normalized_dataset.reshape(-1, 1)

        loss = loss_f(pred, target)

        loss.backward()
        opt.step()

        train_loss += loss.item()
    print("train_loss:{:.6f}".format(train_loss/len(train_loader)))
    return train_loss / len(train_loader)

#test&predict

def test():
    model.eval()
    test_loss = 0

    for idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)

        pred = model(data)
        pred = pred.view(-1)

        loss = loss_f(pred, target)

        test_loss += loss.item()

    print("test_loss:{:.6f}".format(test_loss))

    return test_loss


if __name__ == "__main__":
    train_losses = []
    test_losses = []

    for i in range(100):
        train_loss = train()
        test_loss = test()

        train_losses.append(train_loss)
        test_losses.append(test_loss)

    torch.save(model.state_dict(), "week9/seaborn/model.pt")

    plt.figure(1)
    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.show()
