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
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os

# Define the paths to the "yes" and "no" folders
yes_folder_path = "C:/Users/meena/Downloads/stress/yes"
no_folder_path = "C:/Users/meena/Downloads/stress/no"

# List to store image paths and corresponding labels
image_paths = []
labels = []

# Load images from the "yes" folder (stressed individuals)
yes_image_paths = [os.path.join(yes_folder_path, filename) for filename in os.listdir(yes_folder_path)]
image_paths.extend(yes_image_paths)
labels.extend([1] * len(yes_image_paths))  # Label 1 for stressed individuals

# Load images from the "no" folder (non-stressed individuals)
no_image_paths = [os.path.join(no_folder_path, filename) for filename in os.listdir(no_folder_path)]
image_paths.extend(no_image_paths)
labels.extend([0] * len(no_image_paths))  # Label 0 for non-stressed individuals

# Verify the loaded image paths and labels
print("Image paths:", image_paths)
print("Labels:", labels)

# Check if the dataset contains more than one image
if len(image_paths) <= 1:
    print("Error: Insufficient data. Please ensure your dataset contains more than one image.")
else:
    # Proceed with data processing and classification
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
        
        # Call ML/DL classifiers
        classify_with_cnn(gray_image)
        # classify_with_decision_tree(bilateral_filtered_image)
        # classify_with_svm(lbp_image)

    def classify_with_cnn(image):
        # CNN Model
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image.shape[0], image.shape[1], 1)))  # Add 1 for grayscale images
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        

        # Assuming you have some data to train and test
        X_train, X_test, y_train, y_test = train_test_split(image_paths, labels, test_size=0.2, random_state=42)
        
        # Load images for training and testing
        X_train_data = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in X_train]
        X_test_data = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in X_test]
        
        # Reshape images for CNN input
        X_train_data = np.array(X_train_data).reshape(-1, image.shape[0], image.shape[1], 1)  # Add 1 for grayscale images
        X_test_data = np.array(X_test_data).reshape(-1, image.shape[0], image.shape[1], 1)    # Add 1 for grayscale images
        
        # Train the model
        model.fit(X_train_data, y_train, epochs=10, batch_size=32, validation_data=(X_test_data, y_test))
        
        # Evaluate the model
        loss, accuracy = model.evaluate(X_test_data, y_test)
        print("CNN Accuracy: {:.2f}%".format(accuracy * 100))

    # Define classify_with_decision_tree and classify_with_svm functions similarly

    # Create Tkinter window
    root = tk.Tk()
    root.title("Image Processor")

    # Create a button to choose the image
    choose_button = tk.Button(root, text="Choose Image", command=choose_image)
    choose_button.pack()

    root.mainloop()
