import torch
from torch import nn, optim
from torchvision.datasets import mnist
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from model import Discriminator, Generator

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_data = mnist.MNIST('week5/Data/mnist', train = True, transform = transform, download = False)
test_data = mnist.MNIST('week5/Data/mnist', train = False, transform = transform, download = False)

train_loader = DataLoader(train_data, batch_size = 64, shuffle = True, drop_last = True) 
test_loader = DataLoader(test_data, batch_size = 64, shuffle = True)

'''
for i, (data, target) in enumerate(train_loader):
    if i > 0:
        break
    for j in range(16):
        im = data[j]
        plt.subplot(4, 4, j+1)
        plt.title(target[j].numpy())
        plt.xticks([])
        plt.yticks([])
        plt.imshow(im.numpy().squeeze(), cmap = 'gray')

plt.tight_layout()
plt.show()
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D = Discriminator().to(device)
G = Generator(100).to(device)

loss_f = nn.BCELoss()
opt_D = optim.Adam(D.parameters(), lr = 1e-5)
opt_G = optim.Adam(G.parameters(), lr = 1e-5)

def train(epoch):
    D.train()
    G.train()
    d_losses = 0
    g_losses = 0

    for idx, (data, target) in enumerate(train_loader):
        opt_D.zero_grad()
        x_real = data.to(device)
        y_real = torch.ones(64, ).to(device)
        y_real_pred = D(x_real)
        d_real_loss = loss_f(y_real_pred.view(-1), y_real)
        d_real_loss.backward()

        noise =  torch.randn(64, 100, 1, 1, device = device)
        x_fake = G(noise)
        y_fake = torch.zeros(64, ).to(device)
        y_fake_pred = D(x_fake)
        d_fake_loss = loss_f(y_fake_pred.view(-1), y_fake)
        d_fake_loss.backward()

        opt_D.step()
        d_losses += d_real_loss.item() + d_fake_loss.item()

        opt_G.zero_grad()
        noise = torch.randn(64, 100, 1, 1, device = device)
        x_fake = G(noise)
        y_fake = torch.ones(64, ).to(device)
        y_fake_pred = D(x_fake)
        g_loss = loss_f(y_fake_pred.view(-1), y_fake)
        g_loss.backward()       
        g_losses += g_loss.item()

        opt_G.step()
    
    return d_losses / len(train_loader), g_losses / len(train_loader)

def test(epoch):
    G.eval()
    img = []
    with torch.no_grad():
        noise = torch.randn(64, 100, 1, 1, device = device)
        fake_img = G(noise)
        img.append(make_grid(fake_img, normalize = True, pad_value = 1))

    return img

if __name__ == "__main__":
    d_losses = []
    g_losses = []

    for i in range(10):
        d_loss, g_loss = train(i)
        img = test(i)
        d_losses.append(d_loss)
        g_losses.append(g_loss)
        print("epoch:{} d_loss:{:.6f} g_loss:{:.6f}".format(i+1, d_loss, g_loss))
        torch.save(G.state_dict(), "Generator.pt")
    
    plt.figure(1)
    plt.plot(d_losses, label = 'D_loss')
    plt.plot(g_losses, label = 'G_loss')
    plt.show()   
