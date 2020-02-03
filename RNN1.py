# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 13:06:27 2020

@author: SATWIK RAM K
"""

#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the Training Dataset
train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = train.iloc[:, 1:2].values

#Featuring Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
scaled_training_data = sc.fit_transform(training_set)

#Creating Data Structure with 60 Time Stamps and 1 output
x_train = []
y_train = []
for i in range(60,1258):
    x_train.append(scaled_training_data[i-60:i, 0])
    y_train.append(scaled_training_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshaping
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Building the RNN
#Importing the tensorflow libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

#Initilizing RNN
model = Sequential()

#Adding first LSTM Layer and Dropout Regularization
model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(rate = 0.2))

#Adding Second LSTM Layer and Dropout Regularization
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(rate = 0.2))

#Adding Third LSTM Layer and Dropout Regularization
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(rate = 0.2))

#Adding fourth LSTM Layer and Dropout Regularization
model.add(LSTM(units = 50))
model.add(Dropout(rate = 0.2))

#Adding the output Layer
model.add(Dense(units = 1))

#Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error' )

#Fitting the RNN to the Training Set
model.fit(x_train, y_train, epochs = 100, batch_size = 32)

#Making the predictions and visulization

#Getting the real stock price of google
test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((train['Open'], test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()






