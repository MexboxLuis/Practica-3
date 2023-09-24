import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 
from functions.canny import canny
from functions.otsuToHoughTransform import otsuToHoughTransform

def houghTransform(image, error):

    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot the original image
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # Convert to RGB
    axes[0, 0].set_title('Original image')

    # Define the values of theta (angle) and rho (distance from origin)
    theta_res = 1  # Resolution of theta in degrees
    rho_res = 1    # Resolution of rho in pixels

    # Define theta and rho ranges
    theta = np.deg2rad(np.arange(-90, 90, theta_res))
    height, width, _ = image.shape
    max_rho = int(np.sqrt(height**2 + width**2))
    rho = np.arange(-max_rho, max_rho, rho_res)

    # Create Hough space
    hough_space = np.zeros((2 * max_rho, len(theta)), dtype=np.uint64)

    # Find edges
    edges = canny(image)

    # Plot the edges
    axes[0, 1].imshow(edges, cmap='gray')
    axes[0, 1].set_title('Edges')

    # Get edge coordinates
    y_idx, x_idx = np.where(edges > 0)

    # Calculaten the Hough space
    for i in range(len(x_idx)):
        x = x_idx[i]
        y = y_idx[i]
        for t_idx in range(len(theta)):
            r = int(x * np.cos(theta[t_idx]) + y * np.sin(theta[t_idx]))
            hough_space[r + max_rho, t_idx] += 1
    
    # Get the max and min values of the Hough space
    max_value = int(np.max(hough_space))
    min_value = int(np.min(hough_space))

    # Calculate the threshold with Otsu's method getting the maximun intra-class variance
    threshold = otsuToHoughTransform(hough_space) - (max_value) * error # 0 <= error <= 1

    # Plot the Hough space
    axes[1, 0].imshow(hough_space)
    axes[1, 0].set_title('Hough space (Max value: ' + str(max_value)+ ' Threshold: ' + str(threshold) + ')')
    
    # Get the coordinates of the peaks
    y_peaks, x_peaks = np.where(hough_space > threshold)

    # Create a copy of the original image to draw lines
    imgResult = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB

    # Draw lines in the image
    for i in range(len(x_peaks)):
        r = rho[y_peaks[i]]
        t = theta[x_peaks[i]]
        a = np.cos(t)
        b = np.sin(t)
        x0 = a * r
        y0 = b * r
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(imgResult, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Plot the image with lines
    axes[1, 1].imshow(imgResult)
    axes[1, 1].set_title('Image with lines')

    # Adjust subplot spacing
    plt.tight_layout()

    # Show all subplots
    plt.show()

if __name__ == "__main__":
    folder_path = 'images/hough/images'

    image_names = os.listdir(folder_path)

    for image_name in image_names:
        if image_name.endswith('.png'):
            image_path = os.path.join(folder_path, image_name)
            houghTransform(cv2.imread(image_path), error=0.05)
