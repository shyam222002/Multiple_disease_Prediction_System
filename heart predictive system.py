# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 19:00:14 2023

@author: LUCKY
"""

import numpy as np
import pickle

#loading the saved model
loaded_model = pickle.load(open('C:/Users/LUCKY/Desktop/Multiple Disease Prediction System/trained_model_heart.sav','rb'))

input_data = (58,0,0,100,248,0,0,122,0,1,1,0,2)

# change the input_data into numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == 0):
  print("The person DOES NOT HAVE Heart Disease")
else:
  print("The person HAVE Heart Disease")