import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from model import Generator, Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator(100).to(device)
G.load_state_dict(torch.load("Generator.pt", map_location = torch.device(device)))

img = []
noise = torch.randn(64, 100, 1, 1, device = device)
fake_img = G(noise)
img.append(make_grid(fake_img, normalize = True, pad_value = 1))

fig = plt.figure(dpi = 200)
im = img[0].cpu().detach().numpy().transpose((1, 2, 0))
plt.imshow(im)
plt.xticks([])
plt.yticks([])
plt.show()