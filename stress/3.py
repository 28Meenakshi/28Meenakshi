# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 23:32:03 2024

@author: meena
"""

import tkinter as tk
from tkinter import filedialog
import cv2
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from skimage.feature import local_binary_pattern
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def choose_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        process_image(file_path)

def process_image(file_path):
    # Read the original image
    original_image = cv2.imread(file_path)
    
    # Resize the image
    resized_image = cv2.resize(original_image, (300, 300))
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter
    bilateral_filtered_image = cv2.bilateralFilter(gray_image, 9, 75, 75)
    
    # Feature extraction
    mean_val = np.mean(bilateral_filtered_image)
    std_val = np.std(bilateral_filtered_image)
    radius = 3
    n_points = 8 * radius
    lbp_image = local_binary_pattern(bilateral_filtered_image, n_points, radius, method='uniform')
    
    # Display the original image
    plt.figure()
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    plt.show()
    
    # Display the resized image
    plt.figure()
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    plt.title('Resized Image')
    plt.axis('off')
    plt.show()
    
    # Display the grayscale image
    plt.figure()
    plt.imshow(gray_image, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')
    plt.show()
    
    # Display the processed image
    plt.figure()
    plt.imshow(bilateral_filtered_image, cmap='gray')
    plt.title('Processed Image')
    plt.axis('off')
    plt.show()
    
    print("Mean: ", mean_val)
    print("Standard Deviation: ", std_val)
    print("LBP Image: ", lbp_image)

# Create Tkinter window
root = tk.Tk()
root.title("Image Processor")

# Create a button to choose the image
choose_button = tk.Button(root, text="Choose Image", command=choose_image)
choose_button.pack()

root.mainloop()
