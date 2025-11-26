# 🍄 Mushroom Edibility Predictor

A Machine Learning Web Application built with **Streamlit** that predicts whether a mushroom is **Edible** or **Poisonous** based on its physical characteristics.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)

## 📖 Overview
Mushroom Edibility Predictor is an end-to-end Machine Learning web application built using Python and Streamlit. It solves a binary classification problem to determine if a mushroom is Edible or Poisonous based on physical traits. The app allows users to interactively train and compare three algorithms—Support Vector Machine (SVM), Logistic Regression, and Random Forest—while visualizing real-time metrics like Confusion Matrices and ROC Curves to identify the best-performing model.

## ✨ Features
* **Interactive Sidebar:** Adjust hyperparameters (C, Kernel, Max Depth, etc.) in real-time.
* **Multiple Classifiers:**
    * Support Vector Machine (SVM)
    * Logistic Regression
    * Random Forest
* **Visual Metrics:**
    * Confusion Matrix
    * ROC Curve
    * Precision-Recall Curve
* **Best Model Finder:** A specific feature to run all models simultaneously and determine the winner based on accuracy.
* **Data Overview:** Option to view the raw dataframe.

##Project Structure
├── app.py               # The main Streamlit application code
├── mushrooms.csv        # The dataset file
├── requirements.txt     # List of python dependencies
└── README.md            # Project documentation

##Dataset
The dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family. Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one.
