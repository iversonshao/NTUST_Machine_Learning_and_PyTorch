import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, random_split, DataLoader

scaler = MinMaxScaler(feature_range = (-1, 1))

class temp(Dataset):
    def __init__(self, data):
        self.df = data
        self.org_data = self.df.mean_temp.to_numpy()
        self.normalized_dataset = np.copy(self.org_data)
        self.normalized_dataset = self.normalized_dataset.reshape(-1, 1)
        self.normalized_dataset = scaler.fit_transform(self.normalized_dataset)
        self.normalized_dataset = self.normalized_dataset.reshape(-1, 1) #also reshape(-1)
        self.sample_len = 12
    
    def __len__(self):
        if len(self.org_data) > self.sample_len:
            return len(self.org_data) - self.sample_len
        else:
            return 0
        
    def __getitem__(self, idx):
        target = self.normalized_dataset[idx + self.sample_len]
        target = np.array(target).astype(np.float32)

        input = self.normalized_dataset[idx:(idx + self.sample_len)] #n-1
        input = input.reshape(-1, 1)
        #input = input.astype(np.double)

        input = torch.from_numpy(input).float()
        target = torch.from_numpy(target).float()

        return input, target
    
if __name__ == "__main__":
    df = pd.read_csv("temperature.csv", header = 0)
    dataset = temp(df)

    print(dataset[0])
