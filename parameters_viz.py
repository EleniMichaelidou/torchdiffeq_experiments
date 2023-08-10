import numpy as np
import matplotlib.pyplot as plt

# Model names and their corresponding parameter counts for two datasets
models = ['ResNet', 'ODENet']
parameters_mnist = [576778, 208266]
parameters_cifar10 = [577930, 209418]

# Width of each bar
bar_width = 0.25

# Spacing between groups of bars
spacing = 0.05

# X-axis positions for the bars
indices = np.arange(len(models))

# Create grouped bar plot
plt.bar(indices - (bar_width + spacing) / 2, parameters_mnist, bar_width, label='MNIST')
plt.bar(indices + (bar_width + spacing) / 2, parameters_cifar10, bar_width, label='CIFAR-10')

# Add labels and title
plt.xlabel('Models')
plt.ylabel('Number of Parameters')
plt.title('Number of Parameters for Different Models and Datasets')
plt.xticks(indices, models)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
