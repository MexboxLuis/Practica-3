import numpy as np
import cv2
import matplotlib.pyplot as plt

def otsuToHoughTransform(img): #This function is used to calculate the threshold with Otsu's method getting the maximun intra-class variance (created just for hough transform)

    # Calculate the max and min pixel values
    max_pixel = int(np.max(img))
    min_pixel = int(np.min(img))
    
    # Calculate histogram
    hist = np.histogram(img, bins=max_pixel+1, range=(0, max_pixel))[0]

    # Calculate total number of pixels
    total = np.sum(hist)

    # Calculate sum of all pixel values
    sum_total = np.sum(hist * np.arange(0, max_pixel+1))

    ThresholdList = np.arange(0, max_pixel+1)
    var_intra = np.zeros(max_pixel+1)

    for Threshold in ThresholdList:
        # Calculate background weight
        w_bg = np.sum(hist[0:Threshold]) / total if total > 0 else 0

        # Calculate foreground weight
        w_fg = np.sum(hist[Threshold+1:max_pixel+1]) / total if total > 0 else 0

        # Calculate background mean
        m_bg = (np.sum(hist[0:Threshold] * np.arange(0, Threshold)) / np.sum(hist[0:Threshold])) if np.sum(hist[0:Threshold]) > 0 else 0

        # Calculate foreground mean
        m_fg = (np.sum(hist[Threshold+1:max_pixel+1] * np.arange(Threshold+1, max_pixel+1)) / np.sum(hist[Threshold+1:max_pixel+1])) if np.sum(hist[Threshold+1:max_pixel+1]) > 0 else 0

        # Calculate variance of background
        var_bg = (np.sum(hist[0:Threshold] * (np.arange(0, Threshold) - m_bg)**2) / np.sum(hist[0:Threshold])) if np.sum(hist[0:Threshold]) > 0 else 0

        # Calculate variance of foreground
        var_fg = (np.sum(hist[Threshold+1:max_pixel+1] * (np.arange(Threshold+1, max_pixel+1) - m_fg)**2) / np.sum(hist[Threshold+1:max_pixel+1])) if np.sum(hist[Threshold+1:max_pixel+1]) > 0 else 0

        # Calculate intra-class variance
        var_intra[Threshold] = w_bg * var_bg + w_fg * var_fg    

    # Get the threshold that maximizes the intra-class variance
    Threshold = np.argmax(var_intra)

    return Threshold
