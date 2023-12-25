import torch
import pandas as pd
import numpy as np
from model import LSTM
from dataset import wind, scaler
import matplotlib.pyplot as plt
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM().to(device)
model.load_state_dict(torch.load('M11215075/M11215075_model.pt'))

def pred(data):
    model.eval()

    with torch.no_grad():
        data =data.to(device)
        y = model(data)
        return y
    
df = pd.read_csv("M11215075/data/wind_dataset.csv", header = 0)
dataset = wind(df)
preds = []

for i in range(len(dataset)):
    data, target = dataset[i]
    data = data.view(1, 366, 1)
    pred_wind = pred(data)
    pred_wind = pred_wind.cpu().detach().numpy()

    act_wind = scaler.inverse_transform(pred_wind)

    preds.append(act_wind.item())
result = pd.DataFrame(preds)
result.to_csv('M11215075/pred.csv',header = ['pred'])
DAY = range(0, df.DATE.size)
mean_wind = df.WIND 
plt.figure(1)
plt.plot(DAY, mean_wind, label = "org")
plt.plot(DAY[366:], preds, label = "pred")
plt.legend()
plt.show()