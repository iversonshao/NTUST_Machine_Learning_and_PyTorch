import torch
from torch import nn
from torchsummary import summary

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
                nn.Conv2d(1, 64, 4, 2, 1),
                nn.LeakyReLU(),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.LeakyReLU(),
                nn.Conv2d(128, 256, 3, 2, 1),
                nn.LeakyReLU(),
                nn.Conv2d(256, 1, 4, 2, 0),
                nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x
class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
                        nn.ConvTranspose2d(z_dim, 256, 4, 1, 0, bias = False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(True),
                        nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(True),
                        nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(True),
                        nn.ConvTranspose2d(64, 1, 4, 2, 3, bias = False),
                        nn.Tanh()
        )
    def forward(self, x):
        x = self.model(x)
        return x
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = Discriminator().to(device)
    G = Generator(100).to(device)
    summary(G, (100, 1, 1))
        