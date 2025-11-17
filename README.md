# MUSHROOM_ANALYSIS
Mushroom Classifier Web App 🍄

This is an interactive web application built with Streamlit that uses machine learning to classify mushrooms as edible or poisonous.

The app allows you to choose between three different classification models, tune their hyperparameters in real-time, and instantly see the results and performance metrics.

Core Features

Model Selection: Instantly switch between three popular classification models:

Support Vector Machine (SVM)

Logistic Regression

Random Forest

Hyperparameter Tuning: Use interactive sliders, radio buttons, and number inputs in the sidebar to tune the core hyperparameters for each model (e.g., C for SVM, n_estimators for Random Forest).

Performance Metrics: After classifying, the app immediately reports key metrics:

Accuracy

Precision

Recall

Interactive Visualizations: Select which performance plots you want to see, generated in real-time:

Confusion Matrix

ROC Curve

Precision-Recall Curve

Data Inspection: Includes a checkbox to show the raw, pre-processed mushrooms.csv dataset.

Technology Stack

Web Framework: Streamlit

Data Manipulation: Pandas, NumPy

Machine Learning: Scikit-learn (sklearn)

Data Visualization: Matplotlib (via sklearn.metrics displays)

How to Use

Get the Data:

You must have the mushrooms.csv file in the same directory as the app.py script.

Install Dependencies:

Create and activate a virtual environment.

Install the required libraries:

pip install -r requirements.txt


Run the App:

Open your terminal in the project folder and run:

streamlit run app.py


Interact with the App:

A web browser window will open automatically.

Use the sidebar to choose a classifier.

Adjust the hyperparameters for that model.

Select the metrics you'd like to see plotted.

Click the "Classify" button to train the model, get predictions, and see the results.
