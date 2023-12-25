import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class homeworkData(Dataset):
    def __init__(self, df):
        super(homeworkData, self).__init__()
        self.x = df[:,  0].reshape(-1, 1)
        self.y = df[:, -1].reshape(-1, 1)
    
    def __getitem__(self, index):
    
        return self.x[index], self.y[index]
    
    def __len__(self):
    
        return len(self.x)

if __name__ == "__main__":
    df = pd.read_csv("week4/Data/1,01.csv", header = 0)
    df = df.to_numpy(dtype = np.float32)
    df = torch.from_numpy(df)

    train_data = homeworkData(df)
    print(train_data.x[2])