# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 20:30:00 2019

My own decission tree classifier

@author: Jim
"""

import pandas as pd
import numpy as np
import random
from collections import Counter
from sklearn.model_selection import train_test_split as split
from sklearn.metrics import classification_report
from sklearn import tree as sk_tree


class QualitativeDecisionTree:
    def __init__(self):
        self.question = ""
        self.decisions = {}
        ## Idea: maybe create a depth variable that tells how deep tree is.
        
    def ask(self, data_point):
        # I "ask" the data what it is and make a "decision" for the next action
        answer = data_point[self.question]
        # The test data often gives us stuff we didn't build the tree for
        if answer not in self.decisions:
            # Let's consider asking a different question
            object_decisions = dict(filter(lambda x: 
                str(type(x)) == "<class '__main__.QualitativeDecisionTree'>", 
                self.decisions.items()))
            if len(object_decisions) > 0:
                decision = random.choice(list(object_decisions))
            # But if there are no questions left, just predict the most common
            else:
                decision, count = Counter(
                        self.decisions.values()).most_common(1)[0]
        else:
            decision = self.decisions[answer]
        if type(decision) == type('str'):
            return decision
        else: 
            return decision.ask(data_point)

    # Building a visualization would be nice. This is a quick solution.
    def reveal_structure(self, parent = "root"):
        if parent == "root":
            print("\nRoot node: '{}'".format(self.question))
        else:
            print("Node: Value = '{}', Child = '{}'".format(
                    parent, self.question))
        print("Children of '{}' node:\n".format(self.question))
        for decision in self.decisions:
            if type(self.decisions[decision]) == type('str'):
                print("Leaf: Value = '{}', Classification = '{}'".format(
                        decision,self.decisions[decision]))
            else:
                self.decisions[decision].reveal_structure(decision)
            print("End of '{}' branch\n~~~~~~~~\n".format(decision))

class DT_Classifier:
    def __init__(self):
        self.tree = QualitativeDecisionTree()
        self.full_data = pd.DataFrame()
    
    def get_best_attribute(self, data):
        entropies = {}
        for attribute in data.columns.values[:-1]:
            data_count = len(data)
            entropy = 0
            for value in data[attribute].unique():
                value_count = len(data[data[attribute] == value])
                value_entropy = 0     
                for target in data.iloc[:,-1].unique():
                    fraction = len(data[(data[attribute] == value) & 
                            (data.iloc[:,-1] == target)]) / value_count
                    if fraction == 0:
                        entropy_term = 0.0
                    else:
                        entropy_term = (fraction) * np.log2(fraction)
                    value_entropy -= entropy_term
                entropy += value_entropy * value_count / data_count
            entropies[attribute] = entropy
        # Credit to Alex Martelli, https://stackoverflow.com/questions/3282823/get-the-key-corresponding-to-the-minimum-value-within-a-dictionary
        return(min(entropies, key = entropies.get))        

    def fit(self, data, targets):
        # Combine into one DataFrame()
        self.full_data = pd.concat([data, targets], axis=1)
        # Proceed
        self.tree = self.build_tree(self.full_data)
        
    def build_tree(self, data):
        if len(data.columns.values) == 1:
            # No more "questions"
            return data.iloc[:,-1].mode()[0]
        elif len(data.iloc[:,-1].unique()) == 1:
            # All the values are the same
            return data.iloc[0,-1]
        else:
            # Make a new subtree ("question")
            tree = QualitativeDecisionTree()
            tree.question = self.get_best_attribute(data)
            # Add tree nodes
            for value in data[tree.question].unique():
                tree.decisions[value] = self.build_tree(data[data[
                        tree.question] == value].drop(
                        columns = [tree.question]))
            return(tree)

    def predict(self, data):
        predictions = []
        for i, observation in data.iterrows():
            predictions.append(self.tree.ask(observation))
        return pd.DataFrame(predictions)

"""
This is what the data looks like:
    
mood, sex, color, beard_size
happy, m, blue, long
happy, f, blue, short
sad, m, red, med
sad, m, orange, long
sad, f, orange, long
happy, f, yellow, short
sad, m, yellow, med
happy, f, blue, short
sad, m, blue, med
happy, f, red, long
happy, m, yellow, short
sad, f, blue, long
sad, f, red, long

# Check that entropy for color is right:
5/13*(-2/5*np.log2(2/5)-1/5*np.log2(1/5)-2/5*np.log2(2/5)) + 
3/13*(-1/3*np.log2(1/3)-2/3*np.log2(2/3)) + 2/13*(-2/2*np.log2(2/2)) + 
3/13*(-2/3*np.log2(2/3)-1/3*np.log2(1/3))
"""

data = pd.read_csv("data/goofy_data.txt", skipinitialspace = True)
targets = data["beard_size"]
data = data.drop(columns = ["beard_size"])

data_train, data_test, targets_train, targets_test = split(
        data, targets, test_size = 0.2)

classifier = DT_Classifier()
classifier.fit(data_train, targets_train)
predictions = classifier.predict(data_test)

# Test that it works
#predictions.iloc[0] = "short" # Was used to force a bad prediction
print("Predictions of goofy data:")
for i in range(0, len(targets_test)-1):
    print("Predicted: {}, True: {}".format(predictions.values[i],
          targets_test.values[i]))
    print("Correct = {}".format(predictions.values[i] == 
          targets_test.values[i]))
    
# What does the tree look like?
print("\nTree Structure")
classifier.tree.reveal_structure()
    
# Now for some tougher data:
"""
# I found this data on Kaggle and preprocessed it in R:
# (https://www.kaggle.com/cdc/mortality)

library(tidyverse)
library(readr)
library(forcats)

setwd("C:/Users/Jim/Desktop/DS 450")

# This thing is huge. For now, let's just take the first 1000 samples.
d2015 <- read_csv("data/2015_data.csv", n_max = 1000) %>%
  # Just keeping the attributes I want
  select("education_2003_revision", "sex", "age_recode_12", "marital_status",
         "manner_of_death", "race_recode_5", "hispanic_originrace_recode")

# Customize the column names
colnames(d2015) <- c("Education", "Sex", "Age", "Marital_Status", 
        "Manner_of_Death", "Race", "Hispanic_Origin")

# Fix it up so it will work better for a decision tree
d2015 <- d2015 %>% transmute(Age = fct_recode(Age, "Under 1 year" = "01", 
        "1 - 4 years" = "02", "5 - 14 years" = "03", "15 - 24 years" = "04", 
        "25 - 34 years" = "05", "35 - 44 years" = "06", "45 - 54 years" = "07", 
        "55 - 64 years" = "08", "65 - 74 years" = "09", "75 - 84 years" = "10", 
        "85 years and over" = "11", "Age not stated" = "12"), 
        Sex = fct_recode(Sex, "Male" = "M", "Female" = "F"), 
        Marital_Status = fct_recode(Marital_Status, "Never married, 
        single" = "S", "Married" = "M", "Widowed" = "W", "Divorced" = "D", 
        "Unknown" = "U"), Education = fct_recode(as.character(Education), 
        "8th grade or less" = "1", "9 - 12th grade, no diploma" = "2", 
        "High school graduate or GED completed" = "3", "Some college credit, 
        but no degree" = "4", "Associate degree" = "5", 
        "Bachelor’s degree" = "6", "Master’s degree" = "7", 
        "Doctorate or professional degree" = "8", "Unknown" = "9"), 
        Race = fct_recode(as.character(Race), "Other (Puetro Rico only)" = "0", 
        "White" = "1", "Black" = "2", "American Indian" = "3", 
        "Asian or Pacific Islander" = "4"), Hispanic_Origin = fct_collapse(
        as.character(Hispanic_Origin), Hispanic = c("1","2","3","4","5"), 
        Non_Hispanic = c("6","7","8"), Unknown = c("9")), 
        Manner_of_Death = fct_recode(as.character(Manner_of_Death), 
        "Accident" = "1", "Suicide" = "2", "Homicide" = "3", 
        "Pending or Unknown" = "4", "Pending or Unknown" = "5", 
        "Self-Inflicted" = "6", "Natural" = "7", 
        "Pending or Unknown" = "Blank"), Natural_Cause = fct_collapse(
        Manner_of_Death, "Natural" = c("Natural"), "Not Natural" = c(
        "Accident", "Suicide", "Homicide", "Self-Inflicted"), 
        "Pending or Unknown" = c("Pending or Unknown")))

# Save data
write_csv(d2015, "data/Deaths_2015_Small.csv")
"""

data_2 = pd.read_csv("data/Deaths_2015_Small.csv")
targets_2 = data_2["Natural_Cause"]
data_2 = data_2.drop(columns = ["Natural_Cause", "Manner_of_Death"])

data_2_train, data_2_test, targets_2_train, targets_2_test = split(
        data_2, targets_2, test_size = 0.2)

classifier.fit(pd.DataFrame(data_2_train), pd.DataFrame(targets_2_train))
predictions_2 = classifier.predict(pd.DataFrame(data_2_test))

print("\nPerformance of decision tree predicting type of death")
print(classification_report(targets_2_test, predictions_2))

# Call this list to look at the predictions with my eyes
predictions_and_truth = []
for i in range(0, len(targets_2_test.values)-1):
    predictions_and_truth.append("Predicted: {}, True: {}, Correct: {}".format(
            predictions_2.values[i][0], targets_2_test.values[i], 
            (predictions_2.values[i][0] == targets_2_test.values[i])))

# let's see if it's doing a good job:
# Test against sklearn's decision tree
    
# This classifier uses quantitative variables. Reformating...
def factor_to_int(data):
    data = pd.DataFrame(data)
    for column in data:
        str_values = list(data[column].unique())
        int_values = list(range(len(data[column].unique())))
        value_map = dict(zip(str_values, int_values))
        data[column] = data[column].map(value_map)
    return data

data_sk_train = factor_to_int(data_2_train)
data_sk_test = factor_to_int(data_2_test)
targets_sk_train = targets_2_train
targets_sk_test = targets_2_test

sk_dt = sk_tree.DecisionTreeClassifier()
sk_dt.fit(data_sk_train, targets_sk_train)
sk_predictions = sk_dt.predict(data_sk_test)

print("\nPerformance of sklearn decision tree predicting type of death")
print(classification_report(targets_sk_test, sk_predictions))
print("Mine performs a little better than the DecisionTreeClassifier, but this\
 might be unfair because it treats the atributes numerically and I used label-\
encoding.\n")

print("As would be expected, the root node asks about the age of the individua\
l. print(classifier.tree.question)")