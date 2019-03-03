# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 14:55:00 2019

My Neural Network

@author: Jim
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self):
        self.activations = []
        self.errors = []

class Neural_Layer:
    def __init__(self, m, n):
        self.weights = np.random.rand(m, n + 1) - 0.5 # make more parameters later        
        self.is_output = False
        self.nodes = [Neuron() for x in range(m)]

class Perceptron:
    def __init__(self, threshold = -1):
        self.layers = []
        self.bias = threshold
        self.target_values = []
        self.initialized = False # This can be set to False to start over.
    
    def fit(self, data, targets, hidden_layers = [], threshold = None, 
            learning_rate = 0.2, epoch_count = 1, batch_size = 1):
        data = np.array(data)
        if threshold != None:
            self.bias = threshold
        self.target_values = np.unique(targets)
        
        # Initialize layers
        if self.initialized == False:
            # Number of possible outputs
            layer_sizes = [data.shape[1]] + (hidden_layers) + [
                    len(self.target_values)]
        
            for i in range(len(layer_sizes)):
                if i != len(layer_sizes) - 1:
                    self.layers.append(Neural_Layer(
                            layer_sizes[i + 1], layer_sizes[i]))
                else:
                    self.layers[i - 1].is_output = True
            
            self.initialized = True
        
        # The learning
        end_point = len(data)
        for e in range(epoch_count):
            data_index = 0
            for b in range(batch_size):
                if data_index < end_point:
                    inputs = data[data_index]
                    # Feed forward
                    for layer in self.layers:
                        inputs = np.insert(inputs, 0, self.bias)
                        inputs = np.matmul(layer.weights, inputs)
                        inputs = self.activate(inputs)
                        for i in range(len(layer.nodes)):
                            layer.nodes[i].activations.append(inputs[i])
                    # Backpropagation
                    layer_index = len(self.layers)
                    for layer in reversed(self.layers):
                        layer_index -= 1
                        if layer.is_output == True:
                            for j in range(len(layer.nodes)):
                                activation = layer.nodes[j].activations[b]
                                layer.nodes[j].errors.append(activation * (
                                        1 - activation) * (activation - (
                                                self.target_values == 
                                                      targets[data_index])[j]))
                        else:
                            for j in range(len(layer.nodes)):
                                activation = layer.nodes[j].activations[b]
                                layer.nodes[j].errors.append(activation * (
                                        1 - activation) * self.compute_errors(
                                                layer_index, j + 1, b))
                    data_index += 1
            # Update all weights
            for layer in self.layers:
                for j in range(len(layer.nodes)):
                    layer.weights[j] -= learning_rate * np.mean(
                            layer.nodes[j].errors)
                    # Clear
                    layer.nodes[j].activations = []
                    layer.nodes[j].errors = []
            
    def compute_errors(self, layer_index, j, b):
        out_layer = self.layers[layer_index + 1]
        total_error = 0
        for i in range(len(out_layer.nodes)):
            total_error += out_layer.nodes[i].errors[b]*out_layer.weights[i, j]
        return total_error
    
    def predict(self, data, threshold = None):
        data = np.array(data)
        if threshold != None:
            self.bias = threshold
        predictions = []
        for row in data:
            inputs = row
            for layer in self.layers:
                inputs = np.insert(inputs, 0, self.bias)
                inputs = np.matmul(layer.weights, inputs)
                inputs = self.activate(inputs)
            predictions.append(self.target_values[np.argmax(inputs)])
        return predictions
            
    def activate(self, stimulus):
        return [1/(1 + np.exp(-x)) for x in stimulus]

# Prepare some data
wine = pd.read_csv("data/winemag-data-130k-v2.csv", header = 0, nrows = 5000)[
                 ['country','points','price','variety']].dropna(
        subset=['country'])

wine['price'] = wine['price'].fillna(wine['price'].dropna().median())

wine['variety'] = wine['variety'].map(dict(zip(np.unique(wine['variety']), 
    [i for i in range(len(np.unique(wine['variety'])))])))

# Scaling
price_mean = wine.price.mean()
price_std = wine.price.std()
wine.price = (wine.price - price_mean) / price_std

price_max = max(wine.price)
price_min = min(wine.price)
wine.price = (wine.price - price_min) / (price_max - price_min)

# Yes, one-hot encoding would probably be better.
variety_max = max(wine.variety)
variety_min = min(wine.variety)
wine.variety = (wine.variety - variety_min) / (variety_max - variety_min)

points_max = max(wine.points)
points_min = min(wine.points)
wine.points = (wine.points - points_min) / (points_max - points_min)
    
targets = wine['country'].values
data = wine.drop(columns = ['country']).values

training_data, test_data, training_targets, test_targets = train_test_split(
            data, targets, test_size = 0.25, random_state = 42)

# Train on data
model = Perceptron()
scores1 = []
for t in range(25):
    model.fit(training_data, training_targets, hidden_layers = [6, 6], 
                batch_size = 10, epoch_count = 1, learning_rate = 0.2, 
                threshold = -1)
    scores1.append(np.mean(model.predict(test_data) == test_targets))
    
model = Perceptron()
scores2 = []
for t in range(25):
    model.fit(training_data, training_targets, hidden_layers = [6, 6], 
                batch_size = 10, epoch_count = 1, learning_rate = 0.3, 
                threshold = -1)
    scores2.append(np.mean(model.predict(test_data) == test_targets))
    
model = Perceptron()
scores3 = []
for t in range(25):
    model.fit(training_data, training_targets, hidden_layers = [6, 6], 
                batch_size = 10, epoch_count = 1, learning_rate = 0.4, 
                threshold = -1)
    scores3.append(np.mean(model.predict(test_data) == test_targets))
    
model = Perceptron()
scores4 = []
for t in range(25):
    model.fit(training_data, training_targets, hidden_layers = [6, 6], 
                batch_size = 10, epoch_count = 1, learning_rate = 0.5, 
                threshold = -1)
    scores4.append(np.mean(model.predict(test_data) == test_targets))
    
plt.plot(scores1, label = 'η = 0.2')
plt.plot(scores2, label = 'η = 0.3')
plt.plot(scores3, label = 'η = 0.4')
plt.plot(scores4, label = 'η = 0.5')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title("Predicting Wine Origins with MLP\n3 inputs, two hidden layers each\
 size 6, & 27 outputs\nLearning in batches of 10")
plt.legend()
plt.show()

#data_2 = pd.read_csv("data/Deaths_2015_Small.csv")
#targets_2 = data_2["Natural_Cause"]
#data_2 = data_2.drop(columns = ["Natural_Cause", "Manner_of_Death"])
#
#training_data, test_data, training_targets, test_targets = train_test_split(
#        data_2, targets_2, test_size = 0.2, random_state = 29)
#
#model = Perceptron()
#scores1 = []
#rate = 0.75
#for t in range(25):
#    model.fit(training_data, training_targets, hidden_layers = [8, 4], 
#                batch_size = 10, epoch_count = 1, learning_rate = rate, 
#                threshold = -1)
#    scores1.append(np.mean(model.predict(test_data) == test_targets))
#    rate *= 0.1
#    
#plt.plot(scores1)
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
#plt.title("Predicting Nature of Death with MLP\n3 inputs, two hidden layers of\
# size 8 and 4, & 3 outputs\nLearning in batches of 10")
#plt.legend()
#plt.show()

