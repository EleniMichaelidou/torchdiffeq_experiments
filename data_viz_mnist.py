import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch

# Define a transform to convert images to PyTorch tensors
transform = transforms.Compose([transforms.ToTensor()])

# Load the MNIST dataset using torchvision
dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# Select 9 random samples
random_samples = np.random.choice(len(dataset), 9, replace=False)

# Create a 3x3 grid for displaying the images
fig, axes = plt.subplots(3, 3, figsize=(8, 8))

# Display the random images
for i, ax in enumerate(axes.flat):
    image, label = dataset[random_samples[i]]
    image = image.numpy()[0]  # Extract the tensor and convert to numpy array
    ax.imshow(image, cmap='gray')
    ax.set_title(f"Label: {label}", fontsize=18)
    ax.axis('off')

# Generate a unique filename using timestamp
timestamp = int(time.time())
filename = f'mnist_random_images_{timestamp}.png'

# Ensure "images" folder exists
if not os.path.exists('images'):
    os.makedirs('images')

# Save the plot as an image file in the "images" folder
filename = os.path.join('images', filename)
plt.savefig(filename)

plt.tight_layout()
plt.show()
