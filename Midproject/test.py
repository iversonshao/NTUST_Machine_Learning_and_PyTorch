import torch
from torchvision import transforms
from model import CNN
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
from PIL import Image
import os


"""
# 指定主要資料夾的路徑
main_folder = 'Midproject/PokemonData'  # 將路徑替換為您的主要資料夾路徑
# 使用 os.listdir 取得主要資料夾中的所有內容
contents = os.listdir(main_folder)
# 過濾出子資料夾，排除檔案
subfolders = [f for f in contents if os.path.isdir(os.path.join(main_folder, f))]
# 將子資料夾名稱以單引號括住並使用逗號分隔，然後列印
formatted_subfolders = ', '.join(["'{}'".format(subfolder) for subfolder in subfolders])
print(formatted_subfolders)
"""
label = ['Abra', 'Aerodactyl', 'Alakazam', 'Alolan Sandslash', 'Arbok', 'Arcanine', 'Articuno', 'Beedrill', 'Bellsprout', 'Blastoise', 'Bulbasaur', 'Butterfree', 'Caterpie', 'Chansey', 'Charizard', 'Charmander', 'Charmeleon', 'Clefable', 'Clefairy', 'Cloyster', 'Cubone', 'Dewgong', 'Diglett', 'Ditto', 'Dodrio', 'Doduo', 'Dragonair', 'Dragonite', 'Dratini', 'Drowzee', 'Dugtrio', 'Eevee', 'Ekans', 'Electabuzz', 'Electrode', 'Exeggcute', 'Exeggutor', 'Farfetchd', 'Fearow', 'Flareon', 'Gastly', 'Gengar', 'Geodude', 'Gloom', 'Golbat', 'Goldeen', 'Golduck', 'Golem', 'Graveler', 'Grimer', 'Growlithe', 'Gyarados', 'Haunter', 'Hitmonchan', 'Hitmonlee', 'Horsea', 'Hypno', 'Ivysaur', 'Jigglypuff', 'Jolteon', 'Jynx', 'Kabuto', 'Kabutops', 'Kadabra', 'Kakuna', 'Kangaskhan', 'Kingler', 'Koffing', 'Krabby', 'Lapras', 'Lickitung', 'Machamp', 'Machoke', 'Machop', 'Magikarp', 'Magmar', 'Magnemite', 'Magneton', 'Mankey', 'Marowak', 'Meowth', 'Metapod', 'Mew', 'Mewtwo', 'Moltres', 'MrMime', 'Muk', 'Nidoking', 'Nidoqueen', 'Nidorina', 'Nidorino', 'Ninetales', 'Oddish', 'Omanyte', 'Omastar', 'Onix', 'Paras', 'Parasect', 'Persian', 'Pidgeot', 'Pidgeotto', 'Pidgey', 'Pikachu', 'Pinsir', 'Poliwag', 'Poliwhirl', 'Poliwrath', 'Ponyta', 'Porygon', 'Primeape', 'Psyduck', 'Raichu', 'Rapidash', 'Raticate', 'Rattata', 'Rhydon', 'Rhyhorn', 'Sandshrew', 'Sandslash', 'Scyther', 'Seadra', 'Seaking', 'Seel', 'Shellder', 'Slowbro', 'Slowpoke', 'Snorlax', 'Spearow', 'Squirtle', 'Starmie', 'Staryu', 'Tangela', 'Tauros', 'Tentacool', 'Tentacruel', 'Vaporeon', 'Venomoth', 'Venonat', 'Venusaur', 'Victreebel', 'Vileplume', 'Voltorb', 'Vulpix', 'Wartortle', 'Weedle', 'Weepinbell', 'Weezing', 'Wigglytuff', 'Zapdos', 'Zubat']
label = np.array(label)

transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

model = CNN(150)
model.load_state_dict(torch.load('Midproject/model.pt', map_location = "cuda"))
model.eval()

for i in range(1, 10):
    img = Image.open('Midproject/test/0{}.jpg'.format(i)).convert("RGB")
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
img = Image.open('Midproject/test/10.jpg').convert("RGB")
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