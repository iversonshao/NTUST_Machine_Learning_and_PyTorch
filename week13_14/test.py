import os 
import shutil
import torch
import torchvision.datasets as dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import torchvision.utils as vutils
from model import Generator, Discriminator

batch_size = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

root = r'vangogh2photo'
targetC_path = os.path.join(root, 'custom')
output_path = os.path.join('./', r'output')

if os.path.exists(output_path) == False:
    os.mkdir(output_path)
    print('creat dir:', output_path)
subfolder_path = os.path.join(root, 'subfolder')
os.makedirs(subfolder_path, exist_ok=True)

for image_file in os.listdir(targetC_path):
    src = os.path.join(targetC_path, image_file)
    dest = os.path.join(subfolder_path, image_file)
    shutil.move(src, dest)
#dataC_loader = DataLoader(dataset.ImageFolder(targetC_path, transform = transform), batch_size = batch_size, shuffle = True, num_workers = 2)
dataC_loader = DataLoader(dataset.ImageFolder(subfolder_path, transform=transform), batch_size=batch_size, shuffle=True, num_workers=2)

G_B2A = Generator().to(device)
G_B2A.load_state_dict(torch.load('weights/netG_B2A.pth', map_location = torch.device('cuda')))
G_B2A.eval()

if __name__ == '__main__':

    progress_bar = tqdm(enumerate(dataC_loader), total = len(dataC_loader))
    for i, data in progress_bar:
        real_image  = data[0].to(device)
        fake_image = 0.5 * (G_B2A(real_image).data + 1.0)

        vutils.save_image(fake_image.detach(), f"{output_path}/Fake_{i + 1:04d}.jpg", normalize = True)
        progress_bar.set_description(f"Process images {i + 1} of {len(dataC_loader)}")