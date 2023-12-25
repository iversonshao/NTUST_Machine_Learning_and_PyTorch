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

device = torch.device('cuda')

df = pd.read_csv("week4/Data/1,01.csv", header = 0)
x = df['SAT'].to_numpy()
y = df['GPA'].to_numpy()

df = df.to_numpy(dtype = np.float32)
df = torch.from_numpy(df).to(device)

data = homeworkData(df)

model = regression(1, 1).to(device)
model.load_state_dict(torch.load('regression_99.pt'))
model.eval()

pred = []

for data_x, data_y in data:
    pred.append(model(data_x).item())
print(pred)

plt.figure(1)
plt.plot(x, y, 'ro')
plt.plot(x, pred)
plt.title("SAT/GPA")
plt.xlabel("SAT")
plt.ylabel("GPA")
plt.legend()
plt.show()


