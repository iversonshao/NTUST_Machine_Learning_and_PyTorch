import pandas as pd
import numpy as np
import csv
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, random_split, DataLoader


scaler = MinMaxScaler(feature_range = (-1, 1))
class wind(Dataset):
    def __init__(self, data):
        self.df = data
        self.org_data = self.df.WIND.to_numpy()
        self.normalized_dataset = np.copy(self.org_data)
        self.normalized_dataset = self.normalized_dataset.reshape(-1, 1)
        self.normalized_dataset = scaler.fit_transform(self.normalized_dataset)
        self.normalized_dataset = self.normalized_dataset.reshape(-1, 1)
        self.sample_len = 366 

    def __len__(self):
        if len(self.org_data) > self.sample_len:
            return len(self.org_data) - self.sample_len
        #elif len(self.org_data) > self.sample_len1:
            #return len(self.org_data) - self.sample_len1
        else:
            return 0
        
    def __getitem__(self, idx):
        target = self.normalized_dataset[idx + self.sample_len]
        target = np.array(target).astype(np.float32)
    
        input = self.normalized_dataset[idx:(idx + self.sample_len)]
        input = input.reshape(-1, 1)

        input = torch.from_numpy(input).float()
        target = torch.from_numpy(target).float()

        return input, target
    
if __name__ == "__main__":
    df = pd.read_csv("M11215075/data/wind_dataset.csv", header = 0)
    dataset = wind(df)
    print(len(dataset))
    #print(dataset[0])


