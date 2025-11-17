import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt 

def main():
    st.title("Binary Classification")
    st.sidebar.title("Binary Classification Webapp")
    st.markdown("Are your mushrooms edible or poisonous?")
    st.sidebar.markdown("Are your mushrooms edible or poisonous?")

   
    @st.cache_data
    def load_data():

        try:
            data = pd.read_csv("mushrooms.csv")
        except FileNotFoundError:
            st.error("Error: mushrooms.csv not found.")
            st.info("Please make sure the 'mushrooms.csv' file is in the same directory as this app.py file.")
            return None
            
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

   
    @st.cache_data
    def split(df):
        y = df['type'] 
        x = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list, model, x_test, y_test, class_names):
        
        
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
           
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels=class_names, ax=ax)
        
            st.pyplot(fig)

        if "ROC Curve" in metrics_list:
            st.subheader("ROC Curve")
            fig, ax = plt.subplots()
            RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax)
            st.pyplot(fig)

        if "Precision-Recall Curve" in metrics_list:
            st.subheader("Precision-Recall Curve")
            fig, ax = plt.subplots()
            PrecisionRecallDisplay.from_estimator(model, x_test, y_test, ax=ax)
            st.pyplot(fig)

    df = load_data()
    
   
    if df is None:
        st.stop()
        
    x_train, x_test, y_train, y_test = split(df)
    class_names = ['edible', 'poisonous']
    
    st.sidebar.subheader("Choose classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

    if classifier == "Support Vector Machine (SVM)":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_svm')
        kernel = st.sidebar.radio("Kernel", ('rbf', 'linear'), key='kernel_svm') 
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma_svm') 

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'), key='metrics_svm') # FIX 6: Added unique key

        if st.sidebar.button("Classify", key="classify_svm"): 
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            
            
            st.write("Accuracy: ", round(accuracy, 2))
            
            
            st.write("Precision: ", round(precision_score(y_test, y_pred, pos_label=1), 2))
            st.write("Recall: ", round(recall_score(y_test, y_pred, pos_label=1), 2))
            plot_metrics(metrics, model, x_test, y_test, class_names)

    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum no of iterations", 100, 500, key="max_iter_lr")
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'), key='metrics_lr') # FIX 6: Added unique key

        if st.sidebar.button("Classify", key="classify_lr"): 
            st.subheader("Logistics Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            
          
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", round(precision_score(y_test, y_pred, pos_label=1), 2)) 
            st.write("Recall: ", round(recall_score(y_test, y_pred, pos_label=1), 2)) 
            plot_metrics(metrics, model, x_test, y_test, class_names)

    if classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        
        
        n_estimators = st.sidebar.number_input("Number of trees (n_estimators)", 100, 1000, step=10, key='n_estimators_rf')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth_rf')
        
        bootstrap = st.sidebar.radio("Bootstrap samples?", (True, False), key='bootstrap_rf')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'), key='metrics_rf') # FIX 6: Added unique key

        if st.sidebar.button("Classify", key="classify_rf"): 
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            
            
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", round(precision_score(y_test, y_pred, pos_label=1), 2)) 
            st.write("Recall: ", round(recall_score(y_test, y_pred, pos_label=1), 2)) 
            plot_metrics(metrics, model, x_test, y_test, class_names)

    if st.sidebar.checkbox("Show raw data", False, key='show_raw_data'): 
        st.subheader("Mushrooms DataSet (Classification)")
        st.write(df)

if __name__ == '__main__':
    main()