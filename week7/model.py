import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size = 1, hidden_size = 5, num_layers = 1, batch_first = True)
        self.linear = nn.Linear(50, 1) # 10 * 5 = 50

    def forward(self, x):
        r_out, h_state = self.rnn(x)
        #print(r_out)
        out = r_out.reshape(r_out.shape[0], -1)
        out = self.linear(out)

        return out
    
if __name__ == "__main__":
    model = RNN()