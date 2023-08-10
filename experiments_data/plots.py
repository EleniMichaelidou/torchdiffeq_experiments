import pandas as pd
import matplotlib.pyplot as plt

# List of file paths and corresponding custom labels
file_labels = [('mnist_odenet_False.xlsx', 'RK-Net'),
               ('mnist_odenet_True.xlsx', 'ODE-Net'),
               ('mnist_resnet_False.xlsx', 'ResNet')]  # Replace with your actual file paths and labels

# Create a figure and axis
fig, ax = plt.subplots()

# Read data from each sheet in each file and plot
for file_path, custom_label in file_labels:
    df = pd.read_excel(file_path, sheet_name='percentage_0.1')  # Read specific sheet
    ax.plot(df['Epoch'], df['Test Accuracy'], label=custom_label)  # Plot test accuracy vs. epoch with custom label

# Set labels and title
ax.set_xlabel('Epoch')
ax.set_ylabel('Test Accuracy')
ax.set_title('Test Accuracy vs. Epoch')

# Add a legend at the bottom right
ax.legend(loc='lower right')

# Show the plot
plt.tight_layout()
plt.show()
