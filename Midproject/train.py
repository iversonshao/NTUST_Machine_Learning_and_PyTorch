import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CNN
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

lr = 1e-4

#dataloader
train_path = 'Midproject/train'
valid_path = 'Midproject/valid'

transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = datasets.ImageFolder(train_path, transform = transform)
valid_data = datasets.ImageFolder(valid_path, transform = transform)
train_loader = DataLoader(train_data, batch_size = 32, shuffle = True)
valid_loader = DataLoader(valid_data, batch_size = 32, shuffle = True)

#print("label:", train_data.class_to_idx)

device = torch.device("cuda")
model = CNN(150).to(device)
loss_f = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr = lr)

def train():
    model.train()
    train_loss = 0
    train_acc = 0
    
    for idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        
        opt.zero_grad()
        pred = model(data)
        loss = loss_f(pred, label)

        loss.backward()
        opt.step()
        
        _, y = pred.max(1)
        correct = ( y == label ).sum().item() 
        train_acc += correct / data.shape[0]
        train_loss += loss.item() 

    return train_loss / len(train_loader), train_acc / len(train_loader)

def valid():
    model.eval()
    valid_loss = 0
    for idx, (data, label) in enumerate(valid_loader):
        data, label = data.to(device), label.to(device)

        pred = model(data)
        loss = loss_f(pred, label)
        _, y = pred.max(1)
        valid_loss += loss.item()

    return valid_loss / len(valid_loader)

if __name__ == '__main__':
    train_loss_value = []
    valid_loss_value = []
    for i in range(1000):
        train_loss, train_acc = train()
        valid_loss = valid()
        train_loss_value.append(train_loss)
        valid_loss_value.append(valid_loss)
        print("epoch:{} train_loss: {:.6f}, train_acc: {:.2f}".format(i ,train_loss, train_acc))
        print("valid_loss: {:.6f}".format(valid_loss))

    plt.figure(1)
    plt.plot(train_loss_value, 'r', label = "train loss")
    plt.plot(valid_loss_value, 'b', label = "valid_loss")
    plt.title("loss")
    plt.legend()
    plt.show()
    torch.save(model.state_dict(), 'Midproject/model.pt')