import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import streamlit as st
import numpy as np

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file")

if uploaded_file is not None:
    try:
        dataframe = pd.read_csv(uploaded_file)

        # Data Preprocessing
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

        # Print Logistic Regression Metrics
        st.write("-------------------------------------------")
        st.write(" Performance Metrics ")
        st.write("------------------------------------------")
        st.write()
        st.write(" Accuracy for LR :", result_lr, '%')

        # Save the LR model
        filename_lr = 'stress_lr.pkl'
        pickle.dump(logreg, open(filename_lr, 'wb'))

        # Decision Tree
        dt = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=1, min_samples_leaf=10)
        dt.fit(X_train, y_train)
        y_pred_dt = dt.predict(X_test)
        dt_accuracy = accuracy_score(y_test, y_pred_dt)

        # Print Decision Tree Metrics
        st.write()
        st.write("Decision Tree Accuracy:", dt_accuracy*100)
        st.write()

        # Plot comparison graph
        plt.figure(figsize=(10, 6))
        models = ['Logistic Regression', 'Decision Tree']
        accuracies = [result_lr, dt_accuracy*100]
        sns.barplot(x=models, y=accuracies)
        plt.title('Comparison of Model Accuracies')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 100)
        st.pyplot(plt)

        # Example Input Data
        input_data = (25.41, 94.41, 167)  # Example input data

        # Predict using Logistic Regression
        prediction_lr = logreg.predict(np.array(input_data).reshape(1, -1))

        # Print predictions for Logistic Regression
        st.write("Logistic Regression Prediction:", prediction_lr[0])

        # Mapping predictions to stress levels for Logistic Regression
        if prediction_lr[0] == 0:
            st.write("Stress Level (LR): LOW")
        elif prediction_lr[0] == 1:
            st.write("Stress Level (LR): MEDIUM")
        else:
            st.write("Stress Level (LR): HIGH")

        # Load the LR model
        loaded_model_lr = pickle.load(open(filename_lr, 'rb'))

        def stresslevel_prediction(input_data):
            prediction = loaded_model_lr.predict(np.array(input_data).reshape(1, -1))
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

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
