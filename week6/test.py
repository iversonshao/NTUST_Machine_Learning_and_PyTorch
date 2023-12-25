import torch
from torchvision import transforms
from model import CNN
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
from PIL import Image

label = ['cat', 'dog']
label = np.array(label)
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

model = CNN(2)
model.load_state_dict(torch.load('CNN.pt', map_location = 'cuda'))
model.eval()

img = Image.open('../week6/dog-vs-cat/test1/test/1.jpg').convert("RGB")
data = transform(img)

data = torch.unsqueeze(data, dim = 0)
pred = model(data)
#_, y = pred.max(1)
_, y = torch.max(pred, 1)
#title = label[y.cpu().item()]
title = label[y.cpu().detach().numpy()]

plt.figure(1)
plt.imshow(img)
plt.title(str(title))
plt.show()
