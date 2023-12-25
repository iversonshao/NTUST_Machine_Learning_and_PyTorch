import torch
from torch import nn, optim
from torchsummary import summary
import torch.nn.functional as F
#Generator
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, in_channels, 3),
                nn.InstanceNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, in_channels, 3),
                nn.InstanceNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.main(x)
    
class Generator(nn.Module):
    #input->RGB, output->RGB
    def __init__(self, conv_dim = 64, n_res_blocks = 9):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(3, conv_dim, 7),
                nn.InstanceNorm2d(conv_dim),
                nn.ReLU(inplace=True),

                nn.Conv2d(conv_dim, conv_dim * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(conv_dim * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(conv_dim*2, conv_dim * 4, 3, stride=2, padding=1),
                nn.InstanceNorm2d(conv_dim * 4),
                nn.ReLU(inplace=True),

                ResidualBlock(conv_dim * 4),
                ResidualBlock(conv_dim * 4),
                ResidualBlock(conv_dim * 4),
                ResidualBlock(conv_dim * 4),
                ResidualBlock(conv_dim * 4),
                ResidualBlock(conv_dim * 4),
                ResidualBlock(conv_dim * 4),
                ResidualBlock(conv_dim * 4),
                ResidualBlock(conv_dim * 4),

                nn.ConvTranspose2d(conv_dim*4, conv_dim*2, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(conv_dim*2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(conv_dim*2, conv_dim, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(conv_dim),
                nn.ReLU(inplace=True),

                nn.ReflectionPad2d(3),
                nn.Conv2d(conv_dim, 3, 7),
                nn.Tanh()
            )


    def forward(self, x):
        return self.model(x)

#Discriminator
class Discriminator(nn.Module):
    def __init__(self,conv_dim=32):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, conv_dim, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(conv_dim, conv_dim*2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(conv_dim*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(conv_dim*2, conv_dim*4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(conv_dim*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(conv_dim*4, conv_dim*8, 4, padding=1),
            nn.InstanceNorm2d(conv_dim*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(conv_dim*8, 1, 4, padding=1),
        )

    def forward(self, x):
        x = self.main(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = torch.flatten(x, 1)
        return x
    

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator().to(device)
    D = Discriminator().to(device)
    summary(G, (3, 128, 128), batch_size=2, device = 'cuda' if torch.cuda.is_available() else 'cpu')
    summary(D, (3, 128, 128), batch_size=2, device = 'cuda' if torch.cuda.is_available() else 'cpu')
    
