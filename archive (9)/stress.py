import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np
from tkinter.filedialog import askopenfilename
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Load Input Data
filename = askopenfilename()
dataframe = pd.read_csv(filename)

# Data Preprocessing
dataframe.isnull()
dataframe = dataframe.fillna(0)
label_encoder = LabelEncoder()
dataframe['Humidity'] = label_encoder.fit_transform(dataframe['Humidity']) 
dataframe['Temperature'] = label_encoder.fit_transform(dataframe['Temperature']) 

# Data Splitting
X = dataframe.drop(['Stress_Level'], axis=1)
y = dataframe['Stress_Level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Logistic Regression
logreg = LogisticRegression(solver='lbfgs', C=500)
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)
result_lr = accuracy_score(y_test, y_pred_lr) * 100
#=
print("-------------------------------------------")
print(" Performance Metrics ")
print("------------------------------------------")
print()
print(" Accuracy for LR :", result_lr, '%')

# Save the LR model
import pickle
filename = 'stress_lr.pkl'
pickle.dump(logreg, open(filename, 'wb'))
# Decision Tree
dt = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=1, min_samples_leaf=10)  # Adjust parameters
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)
# Print Metrics
print()
print("Decision Tree Accuracy:", dt_accuracy*100)
print()
print()

import matplotlib.pyplot as plt
import seaborn as sns
# Plot comparison graph
plt.figure(figsize=(10, 6))
models = ['Logistic Regression', 'Decision Tree']
accuracies = [result_lr, dt_accuracy*100]
sns.barplot(x=models, y=accuracies)
plt.title('Comparison of Model Accuracies')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.show()


# Example Input Data
input_data = (25.41, 94.41, 167)  # Example input data

# Convert input data into numpy array
id_np_array = np.asarray(input_data)
id_reshaped = id_np_array.reshape(1, -1)

# Predict using Logistic Regression
prediction_lr = logreg.predict(id_reshaped)

# Predict using Decision Tree
prediction_dt = dt.predict(id_reshaped)

# Print predictions
print("Logistic Regression Prediction:", prediction_lr[0])
#print("Decision Tree Prediction:", prediction_dt[0])

# Mapping predictions to stress levels
if prediction_lr[0] == 0:
    print("Stress Level (LR): LOW")
elif prediction_lr[0] == 1:
    print("Stress Level (LR): MEDIUM")
else:
    print("Stress Level (LR): HIGH")

#if prediction_dt[0] == 0:
 #   print("Stress Level (DT): LOW")
#elif prediction_dt[0] == 1:
 #   print("Stress Level (DT): MEDIUM")
#else:
 #   print("Stress Level (DT): HIGH")
 
import streamlit as st 
 
    
 # Loading the trained mode0l
loaded_model = pickle.load(open(filename, 'rb'))

def stresslevel_prediction(input_data):
        id_np_array = np.asarray(input_data)
        id_reshaped = id_np_array.reshape(1, -1)
        prediction = loaded_model.predict(id_reshaped)
        if prediction[0] == 0:
            return "Stress Level: LOW"
        elif prediction[0] == 1:
            return "Stress Level: MEDIUM"
        else:
            return "Stress Level: HIGH"

def main():
        st.title('STRESS LEVEL PREDICTION WEB APP')
        Humidity = st.text_input('Humidity Value')
        Temperature = st.text_input('Body Temperature')
        Step_count = st.text_input('Number of Steps')
        diagnosis = ''
        if st.button('PREDICT'):
            diagnosis = stresslevel_prediction([Humidity, Temperature, Step_count])
        st.success(diagnosis)

if __name__ == '__main__':
        main()