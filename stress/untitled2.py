import tkinter as tk
from tkinter import filedialog
import cv2
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def load_data():
    yes_images = load_images_from_folder("C:/Users/meena/Downloads/stress/yes")
    no_images = load_images_from_folder("C:/Users/meena/Downloads/stress/no")
    
    X = np.array(yes_images + no_images)
    y = np.array([1] * len(yes_images) + [0] * len(no_images))
    
    return X, y

def preprocess_images(images):
    processed_images = []
    for img in images:
        # Resize the image
        resized_image = cv2.resize(img, (300, 300))
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
        processed_images.append(lbp_image.flatten())  # Flatten the image
    return np.array(processed_images)

def train_random_forest(X_train, y_train):
    # Train Random Forest classifier
    clf_rf = RandomForestClassifier(n_estimators=100)
    clf_rf.fit(X_train, y_train)
    return clf_rf

def train_decision_tree(X_train, y_train):
    # Train Decision Tree classifier
    clf_dt = DecisionTreeClassifier()
    clf_dt.fit(X_train, y_train)
    return clf_dt

def train_cnn(X_train, y_train):
    # Train CNN-2D classifier
    num_classes = len(np.unique(y_train))
    input_shape = X_train.shape[1:]
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    return model

def choose_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        process_image(file_path)

def process_image(file_path):
    # Read the original image
    original_image = cv2.imread(file_path)
    processed_image = preprocess_images([original_image])

    # Perform classification
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate classifiers
    clf_rf = train_random_forest(X_train, y_train)
    clf_dt = train_decision_tree(X_train, y_train)
    model_cnn = train_cnn(X_train, y_train)

    # Display classification results
    print("Random Forest Accuracy: ", clf_rf.score(processed_image, [0]))  # Assuming [0] is the label for this image
    print("Decision Tree Accuracy: ", clf_dt.score(processed_image, [0]))  # Assuming [0] is the label for this image
    cnn_loss, cnn_accuracy = model_cnn.evaluate(X_test, y_test)
    print("CNN-2D Accuracy: ", cnn_accuracy)

# Create Tkinter window
root = tk.Tk()
root.title("Image Processor")

# Create a button to choose the image
choose_button = tk.Button(root, text="Choose Image", command=choose_image)
choose_button.pack()

root.mainloop()
