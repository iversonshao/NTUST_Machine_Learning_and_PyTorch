import os
import shutil
import random

path = 'C:/Users/IDLab/Desktop/NTUST_Machine_Learning&PyTorch/Midproject/PokemonData' 
train_folder = 'C:/Users/IDLab/Desktop/NTUST_Machine_Learning&PyTorch/Midproject/train'
valid_folder = 'C:/Users/IDLab/Desktop/NTUST_Machine_Learning&PyTorch/Midproject/valid'
os.makedirs(train_folder, exist_ok=True)
os.makedirs(valid_folder, exist_ok=True)

for folder_name in os.listdir(path):
    folder_path = os.path.join(path, folder_name)

    if os.path.isdir(folder_path):
        image = os.listdir(folder_path)
        random.shuffle(image)

        split = int(0.8 * len(image))

        train_img = image[:split]
        valid_img = image[split:]
        
        print(f"Folder: {folder_name}")
        print(f"Total Images: {len(image)}")
        print(f"Train Images: {len(train_img)}")
        print(f"Validation Images: {len(valid_img)}")


        for image in train_img:
            src = os.path.join(folder_path, image)
            dest = os.path.join(train_folder, folder_name, image)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy(src, dest)

        for image in valid_img:
            src = os.path.join(folder_path, image)
            dest = os.path.join(valid_folder, folder_name, image)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy(src, dest)
print("Done")