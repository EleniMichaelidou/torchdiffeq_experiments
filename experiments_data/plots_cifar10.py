import pandas as pd
import matplotlib.pyplot as plt
import os

# List of file paths and corresponding custom labels
file_labels = [
    ('cifar10_odenet_False.xlsx', 'RK-Net'),
    ('cifar10_odenet_True.xlsx', 'ODE-Net'),
    ('cifar10_resnet_False.xlsx', 'ResNet')
]  # Replace with your actual file paths and labels

# List of percentage values
percentage_values = [0.25, 0.5, 0.75, 1.0]

# Create a "plots" directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# Loop through each percentage value
for percentage in percentage_values:
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Loop through each file and custom label
    for file_path, custom_label in file_labels:
        # Read data from the corresponding sheet in the file
        df = pd.read_excel(file_path, sheet_name=f'percentage_{percentage}')

        # Plot test accuracy vs. epoch with custom label
        ax.plot(df['Epoch'], df['Test Accuracy'], label=custom_label)

    # Set labels and title
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy')
    ax.set_title(f'Test Accuracy vs. Epoch (Percentage {percentage})')

    # Set y-axis limits from 0 to 1
    ax.set_ylim(0, 1.1)

    # Add a legend at the bottom right
    ax.legend(loc='lower right')

    # Generate the plot filename based on percentage
    plot_filename = os.path.join('plots', f'cifar10_plot_percentage_{percentage}.png')

    # Save the plot in the "plots" directory with the dynamic filename
    plt.savefig(plot_filename)

    # Close the figure to release memory
    plt.close()

print("Plots saved successfully!")