# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 19:12:25 2023

@author: LUCKY
"""

import pickle 
import streamlit as st
from streamlit_option_menu import option_menu

# loading the saved models

diabetes_model = pickle.load(open('C:/Users/LUCKY/Desktop/Multiple Disease Prediction System/trained_model.sav','rb'))

heart_disease_model = pickle.load(open('C:/Users/LUCKY/Desktop/Multiple Disease Prediction System/trained_model_heart.sav','rb'))

parkinsons_model = pickle.load(open('C:/Users/LUCKY/Desktop/Multiple Disease Prediction System/trained_model_parkinson.sav','rb'))


# sidebar for navigation

with st.sidebar:
    selected = option_menu("Multiple Disease Prediction System",
                           
                           ["Diabetes Prediction",
                           "Heart Disease Prediction",
                           "Parkinson's Prediction"],
                           
                           icons = ['activity', 'heart', 'person'],
                           
                           default_index=0)
    
# Diabetes Prediction Page
if(selected == "Diabetes Prediction"):
    
    # page title
    st.title("Diabetes Prediction using ML")
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input("Number of Pregnancies")
    
    with col2:
        Glucose = st.text_input("Glucose Level")
        
    with col3:
        BloodPressure = st.text_input("Blood Pressure Value")
    
    with col1:
        SkinThickness = st.text_input("Skin Thickness Value")
        
    with col2:
        Insulin = st.text_input("Insulin Level")
        
    with col3:
        BMI = st.text_input("BMI Value")
        
    with col1:
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")
        
    with col2:
        Age = st.text_input("Age of the person")
        
    # code for prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if(st.button('Diabetes Test Result')):
       diab_prediction = diabetes_model.predict([[ Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
       
       if(diab_prediction[0] == 0):
           diab_diagnosis = "The Person is DIABETIC"
       else:
           diab_diagnosis = "The Person is NOT DIABETIC"
         
    st.success(diab_diagnosis)
            
           

if(selected == "Heart Disease Prediction"):
    
    # page title
    st.title("Heart Disease Prediction using ML")
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input("Age")
        
    with col2:
        sex = st.text_input("Sex")
    
    with col3:
        cp = st.text_input("Chest Pain Types")
        
    with col1:
        trestbps = st.text_input("Resting Blood Pressure")
        
    with col2:
        chol = st.text_input("Serum Cholestrol in mg/dL")
        
    with col3:
        fbs = st.text_input("Fasting Blood Sugar > 120 mg/dL")
        
    with col1:
        restecg = st.text_input("resting electrocardiographic results (values 0,1,2)")
    
    with col2:
        thalach = st.text_input("maximum heart rate achieved")
        
    with col3:
        exang = st.text_input("exercise induced angina")
        
    with col1:
        oldpeak = st.text_input("oldpeak = ST depression induced by exercise relative to rest")
        
    with col2:
        slope = st.text_input("the slope of the peak exercise ST segment")
        
    with col3:
        ca = st.text_input("number of major vessels (0-3) colored by flourosopy")
    
    with col1:
        thal = st.text_input("thal: 0 = normal; 1 = fixed defect; 2 = reversable defect")
        
    # code for prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if(st.button('Heart Disease Test Result')):
       heart_prediction = heart_disease_model.predict([[ age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
       
       if(heart_prediction[0] == 0):
           heart_diagnosis = "The Person DOES NOT have Heart Disease"
       else:
           heart_diagnosis = "The Person HAVE Heart Disease"
         
    st.success(heart_diagnosis)
    
    
if(selected == "Parkinson's Prediction"):
    
    # page title
    st.title("Parkinson's Prediction using ML")
