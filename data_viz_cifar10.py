import os

import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch
import time

# Load the CIFAR-10 dataset using torchvision
transform = transforms.Compose([transforms.ToTensor()])
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)

# Select 9 random samples
random_samples = np.random.choice(len(dataset), 9, replace=False)

# Create a 3x3 grid for displaying the images
fig, axes = plt.subplots(3, 3, figsize=(8, 8))

# Display the random images
for i, ax in enumerate(axes.flat):
    image, label = dataset[random_samples[i]]
    image = image.numpy().transpose((1, 2, 0))  # Transpose to (height, width, channels)
    ax.imshow(image)
    ax.set_title(f"Label: {label}", fontsize=18)  # Increase font size here
    ax.axis('off')

# Generate a unique filename using timestamp
timestamp = time.strftime("%Y%m%d_%H%M%S")
filename = f'cifar_random_images_{timestamp}.png'

# Ensure "images" folder exists
if not os.path.exists('images'):
    os.makedirs('images')

# Save the plot as an image file in the "images" folder
filename = os.path.join('images', filename)
plt.savefig(filename)

plt.tight_layout()

# Display the plot
plt.show()
