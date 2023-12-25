import torch
from torchvision import transforms
from model import Generator
from PIL import Image
import os

# Set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Paths to your saved models
G_X2Y_path = "G_X2Y.pt"
G_Y2X_path = "G_Y2X.pt"

# Load the models
G_XtoY = Generator(conv_dim=64).to(device)
G_YtoX = Generator(conv_dim=64).to(device)

G_XtoY.load_state_dict(torch.load("G_X2Y.pt"))
G_YtoX.load_state_dict(torch.load("G_Y2X.pt"))

# Put the models on the evaluation mode
G_XtoY.eval()
G_YtoX.eval()

# Transform for the input images
transform = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# Paths to your test images
test_image_paths = [
    "test/image1.jpg",
    "test/image2.jpg",
    "test/image3.jpg",
    "test/image4.jpg",
    "test/image5.jpg",
    # Add more image paths as needed
]

# Generate images
with torch.no_grad():
    for image_path in test_image_paths:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Use the corresponding generator based on your needs
        generated_image = G_XtoY(image_tensor)  # Change to G_YtoX if needed

        # Save the generated image
        
        output_path = os.path.join("Finalproject", os.path.basename(image_path))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        transforms.ToPILImage()(generated_image.squeeze(0).cpu()).save(output_path)

print("Generation completed.")