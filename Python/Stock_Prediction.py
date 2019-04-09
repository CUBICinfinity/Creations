# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 13:58:18 2019
Stock Market Prediction
@author: Calvin Garrett https://github.com/CSG118
Taken and started from
https://www.analyticsvidhya.com/blog/2018/10/predicting-stock-price-machine-learningnd-deep-learning-techniques-python/
"""

# Not all of this section is put to use.

#importing required libraries
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

#for normalizing data
scaler = MinMaxScaler(feature_range=(0, 1))

# Get data
data = pd.read_csv('GOOG_.csv')


#setting index as date
data['Date'] = pd.to_datetime(data.Date,format='%Y-%m-%d')
data.index = data['Date']

#plot
plt.figure(figsize=(16,8))
plt.plot(data['Close'], label='Close Price history')

#creating dataframe
data = data.sort_index(ascending=True, axis=0)

# Is this code necessary?
new_data = pd.DataFrame(index=range(0,len(data)),columns=['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

#setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

"""
My own code below (Jim Greene / CubicInfinity)
Predicts using a MLP and special handling of the data.
"""
# Libraries
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# input_length and input_decay specify dimensions of the input nodes.
input_length = 105 
input_decay = 7
# Index of data to training start at
start = 0


# Predict the last 61 days
end = len(new_data)-60 


# Creating train and test sets
dataset = new_data.values
train = dataset[start:end,0]
test = dataset[end:,0]

current_layer = list(train[0:input_length])
first_values = []


# Initialize the first training values
for k in range(input_length-1): # To experiment later: Don't subtract 1
    new_layer = []
    for i in range(len(current_layer)-1):
    
        """if k == 0: # To experiment. There is much room for improvement: 
        # Take back my decision to remove actual values from input.
        # Add more variables, etc.
            new_layer.append(current_layer[i])
        else:"""
        
        new_layer.append(current_layer[i+1] - current_layer[i])
    current_layer = new_layer
    first_values.append(new_layer)


# Prune values
observation = []
for layer in first_values:
    if len(layer) > (input_decay - 1):
        observation.extend(layer[-input_decay:])
    else:
        observation.extend(layer)


# Rearrange 'observation' for convenience
layers = []
for i in range(input_decay):
    layers.append([])

for j in range(input_decay):
    for i in range(input_length - input_decay + j):
        if (i + 1) > (input_length - input_decay):
            extra = (i + 1) - (input_length - input_decay)
            layers[j].append(observation[input_decay*i+j - int((extra**2)/2 + extra/2)])
        else:
            layers[j].append(observation[input_decay*i+j])

# Build training data
observations = np.concatenate(layers).reshape(1, -1)
targets = np.array([])

for i in range(len(train)-input_length):
    next_value = train[(input_length + i)]
    next_difference = next_value - train[(input_length + i - 1)]
    
    del layers[0]
    for j in range(len(layers)):
        del layers[j][-1]
        
    layers.append([next_difference]*(input_length-1))
    for k in range(input_length-2):
        layers[-1][k+1] = layers[-1][k] - layers[-2][k]
        
    observations = np.append(observations, np.concatenate(layers).reshape(1, -1), axis = 0)
    targets = np.append(targets, np.array([next_difference]), axis = 0)

observations = np.delete(observations, -1, 0)
targets = targets.astype('int')

targets = targets.astype('int')


# Standardize
ob_scaler = StandardScaler().fit(observations)
scaled_observations = ob_scaler.transform(observations)


# Neural network
MLP1 = MLPRegressor(solver='adam', 
                   activation = 'tanh',
                   hidden_layer_sizes=(100, 100), 
                   learning_rate_init = 0.0001, 
                   learning_rate='constant', 
                   alpha=0.0001, 
                   epsilon = 1e-08,
                   max_iter=600,
                   verbose = True,
                   nesterovs_momentum = True,
                   shuffle = True,
                   random_state = 8465)

MLP1.fit(scaled_observations, targets)


# Predict
predictions = []

for i in range(len(test)):
    current_ob = ob_scaler.transform(np.concatenate(layers).reshape(1, -1))
    predicted_change = MLP1.predict(current_ob)[0]
    
    predictions.append(predicted_change)
    
    del layers[0]
    for j in range(len(layers)):
        del layers[j][-1]
    layers.append([predicted_change]*(input_length-1))
    for k in range(input_length-2):
        layers[-1][k+1] = layers[-1][k] - layers[-2][k]

predictions[0] = train[-1] + predictions[0]

for i in range(1,len(predictions)):
    predictions[i] = predictions[i-1] + predictions[i]


# Plotting results
rcParams['figure.figsize'] = 10,5
plt.plot(pd.DataFrame(np.stack((predictions, test), axis = -1), columns=['Predictions', 'Actual']))

plot_train = pd.DataFrame(train)
plot_test = pd.DataFrame(test)
plot_test.index = plot_test.index + len(plot_train)
plot_predict = pd.DataFrame(predictions)
plot_predict.index = plot_test.index
plt.plot(plot_train[-200:])
plt.plot(plot_test)
plt.plot(plot_predict)


"""
# Test the same model, but only predict one day at a time.
# This part doesn't work completely because of the lack of actual values being fed as input.
"""
predictions = []

actual_change = test[0] - train[-1]

for i in range(len(test)):
    current_ob = ob_scaler.transform(np.concatenate(layers).reshape(1, -1))
    predicted_change = MLP1.predict(current_ob)[0]
    if i > 0:
        actual_change = test[i] - test[i-1]
    
    predictions.append(predicted_change)
    
    del layers[0]
    for j in range(len(layers)):
        del layers[j][-1]
    layers.append([actual_change]*(input_length-1))
    for k in range(input_length-2):
        layers[-1][k+1] = layers[-1][k] - layers[-2][k]

predictions[0] = train[-1] + predictions[0]

for i in range(1,len(predictions)):
    predictions[i] = predictions[i-1] + predictions[i]


# Plotting
rcParams['figure.figsize'] = 10,5
plt.plot(pd.DataFrame(np.stack((predictions, test), axis = -1), columns=['Predictions', 'Actual']))

plot_train = pd.DataFrame(train)
plot_test = pd.DataFrame(test)
plot_test.index = plot_test.index + len(plot_train)
plot_predict = pd.DataFrame(predictions)
plot_predict.index = plot_test.index
plt.plot(plot_train)
plt.plot(plot_test)
plt.plot(plot_predict)
