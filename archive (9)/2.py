# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 23:39:28 2024

@author: meena
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import streamlit as st

# Load Input Data (commented out for now as we'll handle input differently in Streamlit)
# filename = askopenfilename()
# dataframe = pd.read_csv(filename)

# Mock DataFrame for testing (remove this when integrating with Streamlit)
dataframe = pd.DataFrame({
    'Humidity': [25, 30, 35, 40, 45],
    'Temperature': [94, 96, 98, 100, 102],
    'Step_count': [150, 200, 250, 300, 350],
    'Stress_Level': ['LOW', 'MEDIUM', 'HIGH', 'MEDIUM', 'LOW']
})

# Data Preprocessing
dataframe = dataframe.fillna(0)
label_encoder = LabelEncoder()
dataframe['Stress_Level'] = label_encoder.fit_transform(dataframe['Stress_Level'])

# Data Splitting
X = dataframe.drop(['Stress_Level'], axis=1)
y = dataframe['Stress_Level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Logistic Regression
logreg = LogisticRegression(solver='lbfgs', C=500)
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)
result_lr = accuracy_score(y_test, y_pred_lr) * 100

# Save the LR model
filename_lr = 'stress_lr.pkl'
pickle.dump(logreg, open(filename_lr, 'wb'))

# Decision Tree
dt = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=1, min_samples_leaf=10)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)

# Plot comparison graph
plt.figure(figsize=(10, 6))
models = ['Logistic Regression', 'Decision Tree']
accuracies = [result_lr, dt_accuracy * 100]
sns.barplot(x=models, y=accuracies)
plt.title('Comparison of Model Accuracies')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
# Save the plot to a file instead of displaying it
plt.savefig('comparison_plot.png')
def stresslevel_prediction(input_data, loaded_model):
    # Convert input data to numeric values
    numeric_data = [float(val) for val in input_data]
    
    # Reshape the numeric data
    id_reshaped = np.asarray(numeric_data).reshape(1, -1)
    
    # Predict using the loaded model
    prediction = loaded_model.predict(id_reshaped)
    
    if prediction[0] == 0:
        return "Stress Level: LOW"
    elif prediction[0] == 1:
        return "Stress Level: MEDIUM"
    else:
        return "Stress Level: HIGH"

def main():
    # Load the trained model
    loaded_model = pickle.load(open(filename_lr, 'rb'))
    
    st.title('STRESS LEVEL PREDICTION WEB APP')
    Humidity = st.text_input('Humidity Value')
    Temperature = st.text_input('Body Temperature')
    Step_count = st.text_input('Number of Steps')
    diagnosis = ''
    if st.button('PREDICT'):
        diagnosis = stresslevel_prediction([Humidity, Temperature, Step_count], loaded_model)
    st.success(diagnosis)

if __name__ == '__main__':
    main()
