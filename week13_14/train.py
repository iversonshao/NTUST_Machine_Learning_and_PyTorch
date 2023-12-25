import torch
from torch import nn, optim
from model import Generator, Discriminator, LamdaLR1
import itertools
import random
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epoch = 1
batch_size = 2
decay_epoch = 10
lr = 2e-3
log_freq = 100
trainA_path = 'week13-14/vangogh2photo/new_trainA'
trainB_path = 'week13-14/vangogh2photo/new_trainB'

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

# To store 50 generated image in a pool and sample from it when it is full
class ReplayBuffer:
    def __init__(self, max_size = 50):
        assert (max_size > 0), "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)
    
fake_A_sample = ReplayBuffer()
fake_B_sample = ReplayBuffer()


D_A = Discriminator().to(device)
D_B = Discriminator().to(device)
G_A2B = Generator().to(device)
G_B2A = Generator().to(device)
G_A2B.apply(weights_init_normal)
G_B2A.apply(weights_init_normal)
D_A.apply(weights_init_normal)
D_B.apply(weights_init_normal)

# Loss function
MSE = nn.MSELoss()
L1 = nn.L1Loss()

# Optimizers

opt_G = torch.optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()), lr = lr, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(itertools.chain(D_A.parameters(), D_B.parameters()), lr = lr, betas=(0.5, 0.999))

# clip_value = 0.005
# for p in itertools.chain(G_A2B.parameters(), G_B2A.parameters(), D_A.parameters(), D_B.parameters()):
#     p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda = LamdaLR1(epoch, 0, decay_epoch).step)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda = LamdaLR1(epoch, 0, decay_epoch).step)



#dataloader
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((286, 286)),
    transforms.RandomCrop((256, 256)), #random crop
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) #(0.5,)-> RGB all are 0.5
    ])  

dataA = datasets.ImageFolder(trainA_path, transform = transform)
dataB = datasets.ImageFolder(trainB_path, transform = transform)
dataloader_A = DataLoader(dataA, shuffle = True, batch_size = batch_size, num_workers = 1)
dataloader_B = DataLoader(dataB, shuffle = True, batch_size = batch_size, num_workers = 1)

total_len = len(dataloader_A) + len(dataloader_B)
torch.multiprocessing.set_start_method('spawn', force=True)
torch.cuda.empty_cache()
if __name__ == "__main__":

    
    for i in range(epoch):
        
        progress_bar = tqdm(enumerate(zip(dataloader_A, dataloader_B)), total = total_len)
        for idx, data in progress_bar:
            #print(f"Processing epoch {i}, iteration {idx+1}")
            #train generator
            real_A = data[0][0].to(device)
            real_B = data[1][0].to(device)

            #train G body
            opt_G.zero_grad()
            fake_A = G_B2A(real_B)
            fake_out_A = D_A(fake_A)
            fake_B = G_A2B(real_A)
            fake_out_B = D_B(fake_B)

            real_label = torch.ones(fake_out_A.size(), dtype = torch.float32).to(device)
            fake_label = torch.zeros(fake_out_A.size(), dtype = torch.float32).to(device)
            adversial_loss_B2A = MSE(fake_out_A, real_label.detach())
            adversial_loss_A2B = MSE(fake_out_B, real_label.detach())
            adv_loss = adversial_loss_B2A + adversial_loss_A2B

            rec_A = G_B2A(fake_B)
            rec_B = G_A2B(fake_A)
            consistency_loss_B2A = L1(rec_A, real_A)
            consistency_loss_A2B = L1(rec_B, real_B)
            rec_loss = consistency_loss_A2B + consistency_loss_B2A

            idt_A = G_B2A(real_A)
            idt_B = G_A2B(real_B)
            idt_loss_A = L1(idt_A, real_A)
            idt_loss_B = L1(idt_B, real_B)
            idt_loss = idt_loss_A + idt_loss_B

            #total loss G
            lambda_rec = 10
            lambda_idt = 5
            loss_G = adv_loss +(rec_loss * lambda_rec) + (idt_loss * lambda_idt)

            loss_G.backward()
            opt_G.step()
        

            #train discriminator
            opt_D.zero_grad()
            real_out_A = D_A(real_A)
            real_out_A_loss = MSE(real_out_A, real_label)
            fake_out_A = D_A(fake_A_sample.push_and_pop(fake_A))
            fake_out_A_loss = MSE(fake_out_A, fake_label)
            loss_D_A = real_out_A_loss + fake_out_A_loss
            
            real_out_B = D_B(real_B)
            real_out_B_loss = MSE(real_out_B, real_label)
            fake_out_B = D_B(fake_B_sample.push_and_pop(fake_B))
            fake_out_B_loss = MSE(fake_out_B, fake_label)
            loss_D_B = real_out_B_loss + fake_out_B_loss

            loss_D = (loss_D_A + loss_D_B) * 0.5
            #loss_G.backward(retain_graph=True)
            loss_D.backward()
            opt_D.step()
            #torch.cuda.empty_cache()

            progress_bar.set_description(
                f"[{epoch}/{i+1}][{idx+1}/{total_len}] "
                f"Loss_D: {(loss_D_A + loss_D_B).item():.4f} "
                f"Loss_G: {loss_G.item():.4f} "
                f"Loss_G_identity: {(idt_loss).iteim():.4f} "
                f"loss_G_GAN: {(adv_loss).item():.4f} "
                f"loss_G_cycle: {(rec_loss).item():.4f}")        

            if i % log_freq == 0:
                    vutils.save_image(real_A, f"week13-14/output/real_A_{epoch}.jpg", normalize=True)
                    vutils.save_image(real_B, f"week13-14/output/real_B_{epoch}.jpg", normalize=True)
                
                    fake_A = ( G_B2A( real_B ).data + 1.0 ) * 0.5
                    fake_B = ( G_A2B( real_A ).data + 1.0 ) * 0.5
                    
                    vutils.save_image(fake_A, f"week13-14/output/fake_A_{epoch}.jpg", normalize=True)
                    vutils.save_image(fake_B, f"week13-14/output/fake_B_{epoch}.jpg", normalize=True)

            
        
        torch.save(G_A2B.state_dict(), f"week13-14/weights/netG_A2B_epoch_{epoch}.pth")
        torch.save(G_B2A.state_dict(), f"week13-14/weights/netG_B2A_epoch_{epoch}.pth")
        torch.save(D_A.state_dict(), f"week13-14/weights/netD_A_epoch_{epoch}.pth")
        torch.save(D_B.state_dict(), f"week13-14/weights/netD_B_epoch_{epoch}.pth")

        lr_scheduler_G.step()
        lr_scheduler_D.step()
    
    torch.save(G_A2B.state_dict(), f"week13-14/weights/netG_A2B.pth")
    torch.save(G_B2A.state_dict(), f"week13-14/weights/netG_B2A.pth")
    torch.save(D_A.state_dict(), f"week13-14/weights/netD_A.pth")
    torch.save(D_B.state_dict(), f"week13-14/weights/netD_B.pth")

