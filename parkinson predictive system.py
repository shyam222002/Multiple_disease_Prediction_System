# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 19:07:09 2023

@author: LUCKY
"""

import numpy as np
import pickle

#loading the saved model
loaded_model = pickle.load(open('C:/Users/LUCKY/Desktop/Multiple Disease Prediction System/trained_model_parkinson.sav','rb'))

input_data = (119.992,157.302,74.997,0.00784,0.00007,0.0037,0.00554,0.01109,0.04374,0.426,0.02182,0.0313,0.02971,0.06545,0.02211,21.033,0.414783,0.815285,-4.813031,0.266482,2.301442,0.284654)

#changing input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == 0):
  print("The person DOES NOT have Parkinson's Disease")
else:
  print("The person HAS Parkinson's Disease")