import torch
import torch.nn as nn
import torch.optim as optim
from model import Generator, Discriminator
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
import numpy as np
import os
from PIL import Image
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BASE_DATASET_PATH = "selfie2anime"
BATCH_SIZE = 16
N_WORKERS = 0

IMG_SIZE = 128
LR = 2e-3
BETA1 = 0.5
BETA2 = 0.999

EPOCHS = 100


class Dataset(torch.utils.data.Dataset):

    def __init__(self, img_dir):
        img_dir = BASE_DATASET_PATH + "/" + img_dir 
        
        path_list = os.listdir(img_dir)
        abspath = os.path.abspath(img_dir) 
        
        self.img_dir = img_dir
        self.img_list = [os.path.join(abspath, path) for path in path_list]

        self.transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), # normalize image between -1 and 1
        ])
        
    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx):
        path = self.img_list[idx]
        img = Image.open(path).convert('RGB')

        img_tensor = self.transform(img)
        return img_tensor
class CycleGAN:

    def __init__(self, g_conv_dim=64, d_conv_dim=64):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

        self.G_XtoY = Generator(conv_dim=g_conv_dim).to(self.device)
        self.G_YtoX = Generator(conv_dim=g_conv_dim).to(self.device)

        self.D_X = Discriminator(conv_dim=d_conv_dim).to(self.device)
        self.D_Y = Discriminator(conv_dim=d_conv_dim).to(self.device)

        print(f"Models running of {self.device}")

    def load_model(self, filename):
        save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
        return torch.load(save_filename)

    def real_mse_loss(self, D_out):
        return torch.mean((D_out-1)**2)


    def fake_mse_loss(self, D_out):
        return torch.mean(D_out**2)


    def cycle_consistency_loss(self, real_img, reconstructed_img, lambda_weight):
        reconstr_loss = torch.mean(torch.abs(real_img - reconstructed_img))
        return lambda_weight*reconstr_loss    

    
    def train_generator(self, optimizers, images_x, images_y):
        # Generator YtoX
        optimizers["g_optim"].zero_grad()

        fake_images_x = self.G_YtoX(images_y)

        d_real_x = self.D_X(fake_images_x)
        g_YtoX_loss = self.real_mse_loss(d_real_x)

        recon_y = self.G_XtoY(fake_images_x)
        recon_y_loss = self.cycle_consistency_loss(images_y, recon_y, lambda_weight=10)


        # Generator XtoY
        fake_images_y = self.G_XtoY(images_x)

        d_real_y = self.D_Y(fake_images_y)
        g_XtoY_loss = self.real_mse_loss(d_real_y)

        recon_x = self.G_YtoX(fake_images_y)
        recon_x_loss = self.cycle_consistency_loss(images_x, recon_x, lambda_weight=10)

        g_total_loss = g_YtoX_loss + g_XtoY_loss + recon_y_loss + recon_x_loss
        g_total_loss.backward()
        optimizers["g_optim"].step()

        return g_total_loss.item()

    
    def train_discriminator(self, optimizers, images_x, images_y):
        # Discriminator x
        optimizers["d_x_optim"].zero_grad()

        d_real_x = self.D_X(images_x)
        d_real_loss_x = self.real_mse_loss(d_real_x)
        
        fake_images_x = self.G_YtoX(images_y)

        d_fake_x = self.D_X(fake_images_x)
        d_fake_loss_x = self.fake_mse_loss(d_fake_x)
        
        d_x_loss = d_real_loss_x + d_fake_loss_x
        d_x_loss.backward()
        optimizers["d_x_optim"].step()


        # Discriminator y
        optimizers["d_y_optim"].zero_grad()
            
        d_real_y = self.D_Y(images_y)
        d_real_loss_x = self.real_mse_loss(d_real_y)
    
        fake_images_y = self.G_XtoY(images_x)

        d_fake_y = self.D_Y(fake_images_y)
        d_fake_loss_y = self.fake_mse_loss(d_fake_y)

        d_y_loss = d_real_loss_x + d_fake_loss_y
        d_y_loss.backward()
        optimizers["d_y_optim"].step()

        return d_x_loss.item(), d_y_loss.item()


    def train(self, optimizers, data_loader_x, data_loader_y, print_every=10, sample_every=100):
        losses = []
        g_total_loss_min = np.Inf
    
        fixed_x = next(iter(data_loader_x))[0].to(self.device)
        fixed_y = next(iter(data_loader_y))[0].to(self.device)

        print(f'Running on {self.device}')
        for epoch in range(EPOCHS):
            for (images_x, images_y) in zip(data_loader_x, data_loader_y):
                images_x, images_y = images_x.to(self.device), images_y.to(self.device)
                
                g_total_loss = self.train_generator(optimizers, images_x, images_y)
                d_x_loss, d_y_loss = self.train_discriminator(optimizers, images_x, images_y)
            if epoch % print_every == 0:
                losses.append((d_x_loss, d_y_loss, g_total_loss))
                print('Epoch [{:5d}/{:5d}] | d_X_loss: {:6.4f} | d_Y_loss: {:6.4f} | g_total_loss: {:6.4f}'.format(
                    epoch, 
                    EPOCHS, 
                    d_x_loss, 
                    d_y_loss, 
                    g_total_loss
                ))
                
            if g_total_loss < g_total_loss_min:
                g_total_loss_min = g_total_loss
                
                torch.save(self.G_XtoY.state_dict(), "G_X2Y.pt")
                torch.save(self.G_YtoX.state_dict(), "G_Y2X.pt")
                
                torch.save(self.D_X.state_dict(), "D_X.pt")
                torch.save(self.D_Y.state_dict(), "D_Y.pt")
                
                print("Models Saved")
        return losses
    
if __name__ == "__main__":
    #train
    X_DATASET = "trainA"
    Y_DATASET = "trainB"
    x_dataset = Dataset(X_DATASET)
    y_dataset = Dataset(Y_DATASET)

    data_loader_x = DataLoader(x_dataset, BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)
    data_loader_y = DataLoader(y_dataset, BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)

    # Model
    cycleGan = CycleGAN()

    # Oprimizer
    g_params = list(cycleGan.G_XtoY.parameters()) + list(cycleGan.G_YtoX.parameters())

    optimizers = {
        "g_optim": optim.Adam(g_params, LR, [BETA1, BETA2]),
        "d_x_optim": optim.Adam(cycleGan.D_X.parameters(), LR, [BETA1, BETA2]),
        "d_y_optim": optim.Adam(cycleGan.D_Y.parameters(), LR, [BETA1, BETA2])
    }

    # Train
    losses = cycleGan.train(optimizers, data_loader_x, data_loader_y, print_every=1) 

fig, ax = plt.subplots(figsize=(12,8))
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator, X', alpha=0.5)
plt.plot(losses.T[1], label='Discriminator, Y', alpha=0.5)
plt.plot(losses.T[2], label='Generators', alpha=0.5)
plt.title("Training Losses")
plt.legend()
plt.show()

# val Dataset
x_dataset = Dataset("testA")
y_dataset = Dataset("testB")

data_loader_x = DataLoader(x_dataset, BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)
data_loader_y = DataLoader(y_dataset, BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)
cycleGan.G_XtoY.load_state_dict(torch.load("G_X2Y.pt"))
cycleGan.G_YtoX.load_state_dict(torch.load("G_Y2X.pt"))
cycleGan.D_X.load_state_dict(torch.load("D_X.pt"))
cycleGan.D_Y.load_state_dict(torch.load("D_Y.pt"))

cycleGan.G_XtoY.eval()
cycleGan.G_YtoX.eval()
cycleGan.D_X.eval()
cycleGan.D_Y.eval()
samples = []

for i in range(12):
    fixed_x = next(iter(data_loader_x))[i].to(cycleGan.device)
    fake_y = cycleGan.G_XtoY(torch.unsqueeze(fixed_x, dim=0))
    samples.extend([fixed_x, torch.squeeze(fake_y, 0)])

fig = plt.figure(figsize=(18, 14))
grid = ImageGrid(fig, 111, nrows_ncols=(5, 4), axes_pad=0.5)


for i, (ax, im) in enumerate(zip(grid, samples)):
    _, w, h = im.size()
    im = im.detach().cpu().numpy()
    im = np.transpose(im, (1, 2, 0))
    
    im = ((im +1)*255 / (2)).astype(np.uint8)
    ax.imshow(im.reshape((w,h,3)))

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    if i%2 == 0: title = "selfie"
    else: title = "anime"

    ax.set_title(title)

plt.show()