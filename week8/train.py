import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from model import ResNet, residualblock
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


lr = 1e-2
batch_size = 32
epoch = 5

#dataset
train_path = 'week6/dog-vs-cat/train'
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5 ), (0.5, 0.5, 0.5))])

data = datasets.ImageFolder(train_path, transform)
train_len = int(len(data) * 0.8)
val_len = len(data) - train_len
train_data, val_data = random_split(data, [train_len, val_len])

train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = True)

#loss Function & Optimizer

device = torch.device("cuda")
model = ResNet(224, residualblock, 2).to(device)
loss_f = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr = lr)

#train

def train():
    model.train()
    train_loss = 0
    train_acc = 0
    for idx, (data, label) in enumerate (train_loader):
        data, label = data.to(device), label.to(device)
        opt.zero_grad()
        pred = model(data)
        loss = loss_f(pred, label)
        loss.backward()
        opt.step()

        train_loss += loss.item()
        _, y = pred.max(1)
        correct = ( y == label ).sum().item()
        acc = correct / data.shape[0]
        train_acc += acc

    return train_loss / len(train_loader), train_acc / len(train_loader)

#val

def val():
    model.eval()
    val_loss = 0
    val_acc = 0
    for idx, (data, label) in enumerate (val_loader):
        data, label = data.to(device), label.to(device)
        pred = model(data)
        loss = loss_f(pred, label)

        val_loss += loss.item()
        _, y = pred.max(1)
        correct = ( y == label ).sum().item()
        acc = correct / data.shape[0]
        val_acc += acc
    
    return val_loss / len(val_loader), val_acc / len(val_loader)

if __name__ == '__main__':
    train_loss_value = []
    train_acc_value = []
    val_loss_value = []
    val_acc_value = []
    for i in range(epoch):
        train_loss, train_acc = train()
        val_loss, val_acc = val()
        train_loss_value.append(train_loss)
        val_loss_value.append(val_loss)
        train_acc_value.append(train_acc)
        val_acc_value.append(val_acc)
        print("epoch:{} train_loss: {:.6f}, train_acc: {:.2f}".format(i ,train_loss, train_acc))
        print("valid_loss: {:.6f}, val_acc: {:.2f}".format(val_loss, val_acc))

    plt.figure(1)
    plt.plot(train_loss_value, 'blue', label = "train loss")
    plt.plot(train_acc_value, 'orange', label = "train acc")
    plt.plot(val_loss_value, 'green', label = "valid_loss")
    plt.plot(val_acc_value, 'red', label = "valid acc")
    plt.xlabel('epoch')
    plt.ylabel('acc/loss')
    plt.legend()
    plt.show()
    torch.save(model.state_dict(), 'ResNet.pt')