import torch
import pandas as pd
import numpy as np
from model import LSTM
from dataset import seaborndataset,scaler
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM().to(device)
model.load_state_dict(torch.load('week9/seaborn/model.pt'))

def pred(data):
    model.eval()

    with torch.no_grad():
        data = data.to(device)
        y = model(data)
        return y
    
df = sns.load_dataset("flights")
dataset = seaborndataset(df)
preds = []

for i in range(len(dataset)):
    data, target = dataset[i]
    data = data.view(1, 12, 1)
    pred_temp = pred(data)
    pred_temp = pred_temp.cpu().detach().numpy()
    #print(pred_temp.reshape(-1))

    act_temp = scaler.inverse_transform(pred_temp) #pred_temp.reshape(-1, 1) if self.normalized_dataset = self.normalized_dataset.reshape(-1)
    print(act_temp)
    preds.append(act_temp.reshape(-1))

months = range(0, df.month.size)
passengers_temp = df.passengers

plt.figure(1)
plt.plot(months, passengers_temp, label = "org")
plt.plot(months[12:], preds, label = "pred")
plt.legend()
plt.show()

