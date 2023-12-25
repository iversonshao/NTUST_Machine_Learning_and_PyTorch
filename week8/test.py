import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from model import ResNet, residualblock
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from PIL import Image
import numpy as np

label = ['cat', 'dog']
label = np.array(label)
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

model = ResNet(224, residualblock, 2)
model.load_state_dict(torch.load('week8/ResNet.pt', map_location = 'cuda'))
model.eval()

img = Image.open('../NTUST_Machine_Learning&PyTorch/week6/dog-vs-cat/test1/test/1.jpg').convert("RGB")
data = transform(img)

data = torch.unsqueeze(data, dim = 0)
pred = model(data)
_, y = torch.max(pred, 1)
title = label[y.cpu().detach().numpy()]

plt.figure(1)
plt.imshow(img)
plt.title(str(title))
plt.show()