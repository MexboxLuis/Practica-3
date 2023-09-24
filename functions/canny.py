import numpy as np
import cv2
import matplotlib.pyplot as plt

def canny(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_smoothed = cv2.GaussianBlur(image, (7, 7), 1)
    sobelx = cv2.Sobel(image_smoothed, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image_smoothed, cv2.CV_64F, 0, 1, ksize=3)   
    mag = np.sqrt(sobelx**2 + sobely**2)
    mag = np.uint8(mag / np.max(mag) * 255) # Normalize to 0-255
    theta = np.arctan2(sobely, sobelx)
    rows, cols = image.shape
    non_max = np.zeros((rows, cols), dtype=np.uint8)
    theta = theta * 180. / np.pi
    theta[theta < 0] += 180
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Horizontal gradient
            if (0 <= theta[i, j] < 22.5) or (157.5 <= theta[i, j] <= 180):
                if (mag[i, j] >= mag[i, j - 1]) and (mag[i, j] >= mag[i, j + 1]):
                    non_max[i, j] = mag[i, j]
            # Diagonal gradient
            elif (22.5 <= theta[i, j] < 67.5):
                if (mag[i, j] >= mag[i - 1, j - 1]) and (mag[i, j] >= mag[i + 1, j + 1]):
                    non_max[i, j] = mag[i, j]
            # Vertical gradient
            elif (67.5 <= theta[i, j] < 112.5):
                if (mag[i, j] >= mag[i - 1, j]) and (mag[i, j] >= mag[i + 1, j]):
                    non_max[i, j] = mag[i, j]
            # Diagonal gradient
            elif (112.5 <= theta[i, j] < 157.5):
                if (mag[i, j] >= mag[i + 1, j - 1]) and (mag[i, j] >= mag[i - 1, j + 1]):
                    non_max[i, j] = mag[i, j]

    high_threshold = 0.2 * np.max(non_max)
    low_threshold = 0.1 * np.max(non_max)
    strong_edges = np.zeros((rows, cols), dtype=np.uint8)
    weak_edges = np.zeros((rows, cols), dtype=np.uint8)
    strong_edges[non_max >= high_threshold] = 255
    weak_edges[(non_max <= high_threshold) & (non_max >= low_threshold)] = 255
    final_edges = np.zeros((rows, cols), dtype=np.uint8)
    final_edges[strong_edges == 255] = 255

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if weak_edges[i, j] == 255:
                if 255 in [strong_edges[i - 1, j - 1], strong_edges[i - 1, j], strong_edges[i - 1, j + 1],
                        strong_edges[i, j - 1], strong_edges[i, j + 1],
                        strong_edges[i + 1, j - 1], strong_edges[i + 1, j], strong_edges[i + 1, j + 1]]:
                    final_edges[i, j] = 255
    
    return final_edges
