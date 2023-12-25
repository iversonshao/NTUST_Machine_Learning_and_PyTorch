import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from M11215075_model import CNN
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

lr = 1e-4

#dataloader
train_path = 'Midexam/train'
valid_path = 'Midexam/valid'

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_data = datasets.ImageFolder(train_path, transform = transform)
valid_data = datasets.ImageFolder(valid_path, transform = transform)
train_loader = DataLoader(train_data, batch_size = 16, shuffle = True)
valid_loader = DataLoader(valid_data, batch_size = 2, shuffle =True)

print("label:", train_data.class_to_idx)

#Loss Function & Optimizer
device = torch.device("cuda")
model = CNN(10).to(device)
loss_f = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr = lr)
epoch = 15
def train(epoch):
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
        torch.save(model.state_dict(), 'M11215075_model.pt')
    print("train_epoch:{}, train_loss: {:.6f}, train_acc: {:.2f}".format(epoch + 1, train_loss / len(train_loader), train_acc / len(train_loader)))
    
    return train_loss / len(train_loader), train_acc / len(train_loader)

def valid(epoch):
    model.eval()
    valid_loss = 0
    for idx, (data, label) in enumerate(valid_loader):
        data, label = data.to(device), label.to(device)

        pred = model(data)
        loss = loss_f(pred, label)
        _, y = pred.max(1)
        valid_loss += loss.item()
    print("valid_epoch:{}, valid_loss: {:.6f}".format(epoch + 1, valid_loss/len(valid_loader)))
    
    return valid_loss/len(valid_loader)

if __name__ == '__main__':
    train_loss = []
    valid_loss = []
    for i in range(epoch):
        train_loss.append(train(i))
        valid_loss.append(valid(i))
        
    plt.figure(1)
    plt.plot(train_loss, 'r', label = "train loss")
    plt.plot(valid_loss, 'b', label = "valid_loss")
    plt.title("loss")
    plt.legend()
    plt.show()