import torch
from torch.utils.data import Dataset, random_split, DataLoader
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1,1))

class seaborndataset(Dataset):
    def __init__(self, data):
        self.df = data
        self.org_data = self.df.passengers.to_numpy()
        self.normalized_dataset = np.copy(self.org_data)
        self.normalized_dataset = self.normalized_dataset.reshape(-1, 1)
        self.normalized_dataset = scaler.fit_transform(self.normalized_dataset)
        self.normalized_dataset = self.normalized_dataset.reshape(-1, 1) #also reshape(-1)
        self.sample_len = 12
    
    def __len__(self):
        if len(self.org_data) >= self.sample_len:
            return len(self.org_data) - self.sample_len
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
    df = sns.load_dataset("flights")
    #print(df.head())

    dataset = seaborndataset(df)
    #print(dataset[0])
    #print(dataset.normalized_dataset.shape)
