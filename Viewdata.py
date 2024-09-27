import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from Plotting import Normalize
import utilities as util

def update_image(selected_endmembers, shape, endmembers, abundances, ax):
    # Get the selected endmembers
    selected_indices = [i for i, selected in enumerate(selected_endmembers) if selected]
    
    if len(selected_indices) == 0:
        # If no endmember is selected, show a blank image
        image = np.zeros(shape=(shape[0], shape[1]))
    else:
        # Extract the selected endmembers and compute the RGB representation
        selected_endmembers_data = endmembers[:, selected_indices]  # (num_bands, num_selected_endmembers)
        selected_abundances = abundances[selected_indices, :]  # (num_selected_endmembers, num_pixels)
        
        # Multiply abundances by the corresponding endmembers and sum over the selected endmembers
        image = np.matmul(selected_endmembers_data, selected_abundances)  # (num_bands, num_pixels)
    
    # Normalize the image
    image = image
    
    # Reshape the flat image for visualization (assuming we reshape to a square image)
    image_reshaped = image.T.reshape(shape[0], shape[1], endmembers.shape[0])  # (height, width, num_bands)
    
    # Load the RGB mask and map to spectral response
    rgb_mask = np.loadtxt("RGB_mask.txt")
    spectral_response_matrix = util.map_mask_to_bands(rgb_mask[0:700, :], 112)
    
    # Compute the RGB representation
    rgb_representation = Normalize(np.matmul(image_reshaped, spectral_response_matrix.T))  # (height, width, 3)

    # Update the plot
    ax.clear()
    ax.imshow(rgb_representation)
    ax.axis('off')
    plt.draw()

def visualize(endmembers, abundances, shape):
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.35, right=0.95, top=0.95, bottom=0.05)  # Adjust space for checkboxes and mini-plots

    # Initialize all checkboxes as selected (True)
    num_endmembers = endmembers.shape[1]
    endmember_labels = [f'{i}' for i in range(num_endmembers)]
    selected_endmembers = [True] * num_endmembers

    # Create the CheckButtons widget
    # We'll adjust this to create horizontal alignment of each checkbox with its mini plot
    checkbox_axes = []
    mini_plots = []
    
    # Create a horizontal row for each checkbox and its corresponding mini plot
    for i in range(num_endmembers):
        # Position the checkbox
        rax = plt.axes([0.05, 0.85 - i * 0.08, 0.1, 0.05])  # Adjust positions for each checkbox
        check = CheckButtons(rax, [endmember_labels[i]], [selected_endmembers[i]])
        checkbox_axes.append(check)
        
        # Position the corresponding mini plot
        mini_ax = plt.axes([0.2, 0.85 - i * 0.08, 0.1, 0.05])  # Position horizontally aligned to the checkbox
        mini_plots.append(mini_ax)
        mini_ax.set_xticks([])
        mini_ax.set_yticks([])

        # Plot spectral signature or mini RGB (example: just plotting the spectral signature here)
        mini_ax.plot(endmembers[:, i], color='blue')

    # Define a callback function for checkbox interaction
    def on_checkbox_change(label):
        # Toggle the selected state of the clicked checkbox
        index = endmember_labels.index(label)
        selected_endmembers[index] = not selected_endmembers[index]
        
        # Update the image
        update_image(selected_endmembers, shape, endmembers, abundances, ax)

    # Register the callback function for each checkbox
    for check in checkbox_axes:
        check.on_clicked(on_checkbox_change)

    # Show initial image with all endmembers selected
    update_image(selected_endmembers, shape, endmembers, abundances, ax)
    
    plt.show()