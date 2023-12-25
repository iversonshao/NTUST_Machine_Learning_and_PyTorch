import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size = 1, hidden_size = 500, num_layers = 3, dropout = 0.1, batch_first = True)
        self.linear = nn.Linear(500, 1)

    def forward(self, x):
        h_0 = torch.zeros([3, x.shape[0], 500], device = x.device)
        c_0 = torch.zeros([3, x.shape[0], 500], device = x.device)

        out, _ = self.lstm(x, (h_0.detach(), c_0.detach()))
        out = self.linear(out[:, -1, :])

        return out
    
if __name__ == "__main__": 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = LSTM() 
    print(model)