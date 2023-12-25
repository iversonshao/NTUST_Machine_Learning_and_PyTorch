import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CNN


lr = 1e-4

#dataloader
train_path = 'dog-vs-cat/train'
test_path = 'dog-vs-cat/test1'

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = datasets.ImageFolder(train_path, transform = transform) #讀此路徑下的資料夾
test_data = datasets.ImageFolder(test_path, transform = transform)
train_loader = DataLoader(train_data, batch_size = 128, shuffle = True)
test_loader = DataLoader(test_data, batch_size = 32, shuffle = True)

#print("label:", train_data.class_to_idx)
#loss Function & Optimizer

device = torch.device("cuda")
model = CNN(2).to(device)
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
        train_loss += loss.item() #會溢出

    return train_loss / len(train_loader), train_acc / len(train_loader)

def val():
    val_loss = 0
    model.eval()

    for index, (data, label) in enumerate(test_loader):
        data, label = data.to(device), label.to(device)
        pred = model(data)
        loss = loss_f(pred, label)

        val_loss += loss.item()

    return val_loss / len(test_loader)



if __name__ == '__main__':
    train_loss_value = []
    train_acc_value = []
    val_loss_value = []
    for i in range(2):
        train_loss, train_acc = train()
        val_loss = val()
        train_loss_value.append(train_loss)
        train_acc_value.append(train_acc)
        val_loss_value.append(val_loss)
        print("train_loss: {:.6f}, acc: {:.2f}".format(train_loss, train_acc))
        print("val_loss: {:.6f}".format(val_loss))

    
    torch.save(model.state_dict(), 'model.pt')

