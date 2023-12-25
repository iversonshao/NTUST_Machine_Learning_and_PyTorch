import torch
from torchvision import transforms
from M11215075_model import CNN
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
from PIL import Image

label = ['AFRICAN LEOPARD', 'CARACAL', 'CHEETAH', 'CLOUDED LEOPARD', 'JAGUAR', 'LIONS', 'OCELOT', 'PUMA', 'SNOW LEOPARD', 'TIGER']
label = np.array(label)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

model = CNN(10)
model.load_state_dict(torch.load('M11215075/M11215075_model.pt', map_location = "cuda"))
model.eval()
for i in range(1, 10):
    img = Image.open('test_10/0{}.jpg'.format(i)).convert("RGB")
    data = transform(img)

    data = torch.unsqueeze(data, dim = 0)
    pred = model(data)
    _, y = torch.max(pred, 1)

    title = label[y.cpu().detach().numpy()]
    plt.figure(1)
    plt.imshow(img)
    plt.title(str(title))
    plt.savefig("output_{}.png".format(i))
    plt.show()
    

img = Image.open('test_10/10.jpg').convert("RGB")
data = transform(img)

data = torch.unsqueeze(data, dim = 0)
pred = model(data)
_, y = torch.max(pred, 1)

title = label[y.cpu().detach().numpy()]
plt.figure(1)
plt.imshow(img)
plt.title(str(title))
plt.savefig("output_10.png")
plt.show()
