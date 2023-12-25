import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
import math

#dataset
class timeseries(Dataset):
    def __init__(self, input, output):
        self.x = torch.tensor(input, dtype = torch.float32)
        self.y = torch.tensor(output, dtype = torch.float32)      
        self.len = input.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def __len__(self):
        return self.len
    
if __name__ == '__main__':
    a = np.arange(1, 721, 1)
    b = np.sin(a * np.pi/180) + np.random.randn(720) * 0.05
    b = b.reshape(-1, 1) #turn straight to 1D array
    
    x = []
    y = []
    for i in range(710):
        input = []
        for j in range(i, i+10):
            input.append(b[j])
        x.append(input) #10 time per input
        y.append(b[j+1]) # 1 time per input

    X = np.array(x)
    Y = np.array(y)

    plt.figure(1)
    plt.plot(Y)
    plt.show()
    #plt.figure(2)
    #plt.plot(X)
    #plt.show()

    train_x = X[:360]
    train_y = Y[:360]
    #test_x = X[360:]
    #test_y = Y[360:]
    dataset = timeseries(train_x, train_y)
    print(dataset[0])
    #trainloader = DataLoader(dataset, batch_size = 64, shuffle = True)