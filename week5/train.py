import torch
from torch import nn, optim
from torchvision.datasets import mnist
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from model import DNN
import numpy

lr = 1e-3
epoch = 10

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_data = mnist.MNIST('week5/Data/mnist', train = True, transform = transform, download = False) #不確定是否有資料download = True (第一運行設true確保被下載)
test_data = mnist.MNIST('week5/Data/mnist', train = False, transform = transform, download = False)

train_loader = DataLoader(train_data, batch_size = 128, shuffle = True) #shuffle打亂資料
test_loader = DataLoader(test_data, batch_size = 128, shuffle = True)
"""
plt.figure(1)
image = train_data.data[1].cpu().numpy()
plt.imshow(image, cmap = 'gray')
plt.show()
"""

#Loss & Optimizer
device = torch.device('cuda')
model = DNN(784, 10).to(device)
loss_f = nn.CrossEntropyLoss() #分類用Cross entropy做Loss Function比較好用
opt = optim.Adam(model.parameters(), lr = lr) #之後常用Adam做optimizer

def train():
    train_loss = 0
    train_acc = 0
    model.train()

    for idx, (im, label) in enumerate (train_loader):
        #im.reshape((-1,))
        #print(im)
        im, label = im.to(device), label.to(device)
        opt.zero_grad()
        
        pred = model(im)
        loss = loss_f(pred, label)
        loss.backward()
        opt.step()

        train_loss += loss
        _, y = pred.max(1) #_是機率最大值
        correct = (y == label).sum().item() #pytorch內建生成的label
        acc = correct / im.shape[0]
        train_acc += acc

    return train_loss/len(train_loader), train_acc/len(train_loader)

def test():
    test_loss = 0
    test_acc = 0
    model.eval()

    for idx, (im, label) in enumerate (test_loader):
        #im.reshape((-1,))
        im, label = im.to(device), label.to(device)
        pred = model(im)
        loss = loss_f(pred, label)

        test_loss += loss.item()
        _, y = pred.max(1)

        correct = (y == label).sum().item()
        acc = correct / im.shape[0]
        test_acc += acc

    return test_loss/len(test_loader), test_acc/len(test_loader)

if __name__ == "__main__":

    for i in range(epoch):
        train_loss, train_acc = train()
        test_loss, test_acc = test()

        print("epoch:{}, train_loss:{:.6f},train_acc:{:.4f}, test_loss:{:.6f} , test_acc:{:.4f}".format(i+1, train_loss, train_acc, test_loss, test_acc))
            