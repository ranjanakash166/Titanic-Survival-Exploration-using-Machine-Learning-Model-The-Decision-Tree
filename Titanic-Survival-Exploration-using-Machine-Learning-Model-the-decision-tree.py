
# coding: utf-8

# # Project: Titanic Survival Exploration
# In 1912, the ship RMS Titanic struck an iceberg on its maiden voyage and sank, resulting in the deaths of most of its passengers and crew. In this introductory project, I will explore a subset of the RMS Titanic passenger manifest to determine which features best predict whether someone survived or did not survive. To complete this project, i will need to implement several conditional predictions and answer the questions below.

# # Getting Started
# To begin working with the RMS Titanic passenger data, I'll first need to import the functionality i need, and load our data into a pandas DataFrame.

# In[42]:


import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames

#  for Pretty display for notebooks
get_ipython().magic('matplotlib inline')

# Loading  the dataset
in_file = 'C:/Users/USER/Desktop/titanic_survival_exploration/titanic_data.csv'
full_data = pd.read_csv(in_file)

# Printing the first few entries of the RMS Titanic data
display(full_data.head())


# From a sample of the RMS Titanic data, we can see the various features present for each passenger on the ship:
# 
# Survived: Outcome of survival (0 = No; 1 = Yes)
# 
# Pclass: Socio-economic class (1 = Upper class; 2 = Middle class; 3 = Lower class)
# 
# Name: Name of passenger
# 
# Sex: Sex of the passenger
# 
# Age: Age of the passenger (Some entries contain NaN)
# 
# SibSp: Number of siblings and spouses of the passenger aboard
# 
# Parch: Number of parents and children of the passenger aboard
# 
# Ticket: Ticket number of the passenger
# 
# Fare: Fare paid by the passenger
# 
# Cabin Cabin number of the passenger (Some entries contain NaN)
# 
# Embarked: Port of embarkation of the passenger (C = Cherbourg; Q = Queenstown; S = Southampton)
# 
# Since we're interested in the outcome of survival for each passenger or crew member, we can remove the Survived feature from this dataset and store it as its own separate variable outcomes. We will use these outcomes as our prediction targets.
# 
# Runing the code cell below to remove Survived as a feature of the dataset and storing it in outcomes.
# 

# In[43]:

# Storing the 'Survived' feature in a new variable and remove it from the dataset
outcomes = full_data['Survived']
data = full_data.drop('Survived', axis = 1)

# Showing the new dataset with 'Survived' removed
display(data.head())


# The very same sample of the RMS Titanic data now shows the Survived feature removed from the DataFrame.
# Note that data (the passenger data) and outcomes (the outcomes of survival) are now paired.
# That means for any passenger data.loc[i], they have the survival outcome outcomes[i].
# To measure the performance of my predictions, I need a metric to score my predictions against the true outcomes of survival. Since I am  interested in how accurate my predictions are, I will calculate the proportion of passengers where my prediction of their survival is correct. Running the code cell below to create my accuracy_score function and test a prediction on the first five passengers.
# # Think: Out of the first five passengers, if we predict that all of them survived, what would you expect the accuracy of our predictions to be?

# In[7]:


def accuracy_score(truth, pred):
    """ Returns accuracy score for input truth and predictions. """
    
    # Ensure that the number of predictions matches number of outcomes
    if len(truth) == len(pred): 
        
        # Calculate and return the accuracy as a percent
        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean()*100)
    
    else:
        return "Number of predictions does not match number of outcomes!"
    
# Test the 'accuracy_score' function
predictions = pd.Series(np.ones(5, dtype = int))
print (accuracy_score(outcomes[:5], predictions))


# # Making Predictions
# If I was asked to make a prediction about any passenger aboard the RMS Titanic whom we knew nothing about, then the best prediction i could make would be that they did not survive. This is because i can assume that a majority of the passengers (more than 50%) did not survive the ship sinking.
# The predictions_0 function below will always predict that a passenger did not survive.
# 

# In[8]:

def predictions_0(data):
    """ Model with no features. Always predicts a passenger did not survive. """

    predictions = []
    for _, passenger in data.iterrows():
        
        # Predict the survival of 'passenger'
        predictions.append(0)
    
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_0(data)


# # Question 1
# Using the RMS Titanic data, how accurate would a prediction be that none of the passengers survived?
# 

# In[10]:

print (accuracy_score(outcomes, predictions))


# In[33]:


import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def filter_data(data, condition):
    """
    Removing elements that do not match the condition provided.
    Takes a data list as input and returns a filtered list.
    Conditions should be a list of strings of the following format:
      '<field> <op> <value>'
    where the following operations are valid: >, <, >=, <=, ==, !=
    
    Example: ["Sex == 'male'", 'Age < 18']
    """

    field, op, value = condition.split(" ")
    
    # convert value into number or strip excess quotes if string
    try:
        value = float(value)
    except:
        value = value.strip("\'\"")
    
    # get booleans for filtering
    if op == ">":
        matches = data[field] > value
    elif op == "<":
        matches = data[field] < value
    elif op == ">=":
        matches = data[field] >= value
    elif op == "<=":
        matches = data[field] <= value
    elif op == "==":
        matches = data[field] == value
    elif op == "!=":
        matches = data[field] != value
    else: # catch invalid operation codes
        raise Exception("Invalid comparison operator. Only >, <, >=, <=, ==, != allowed.")
    
    # filter data and outcomes
    data = data[matches].reset_index(drop = True)
    return data

def survival_stats(data, outcomes, key, filters = []):
    """
    Print out selected statistics regarding survival, given a feature of
    interest and any number of filters (including no filters)
    """
    
    # Check that the key exists
   # if key not in data.columns.values :
       # print "'{}' is not a feature of the Titanic data. Did you spell something wrong?".format(key)
       # return False

    # Return the function before visualizing if 'Cabin' or 'Ticket'
    # is selected: too many unique categories to display
    #if(key == 'Cabin' or key == 'PassengerId' or key == 'Ticket'):
      #  print "'{}' has too many unique categories to display! Try a different feature.".format(key)
       # return False

    # Merge data and outcomes into single dataframe
    all_data = pd.concat([data, outcomes], axis = 1)
    
    # Apply filters to data
    for condition in filters:
        all_data = filter_data(all_data, condition)

    # Create outcomes DataFrame
    all_data = all_data[[key, 'Survived']]
    
    # Create plotting figure
    plt.figure(figsize=(8,6))

    # 'Numerical' features
    if(key == 'Age' or key == 'Fare'):
        
        # Remove NaN values from Age data
        all_data = all_data[~np.isnan(all_data[key])]
        
        # Divide the range of data into bins and count survival rates
        min_value = all_data[key].min()
        max_value = all_data[key].max()
        value_range = max_value - min_value

        # 'Fares' has larger range of values than 'Age' so create more bins
        if(key == 'Fare'):
            bins = np.arange(0, all_data['Fare'].max() + 20, 20)
        if(key == 'Age'):
            bins = np.arange(0, all_data['Age'].max() + 10, 10)
        
        # Overlay each bin's survival rates
        nonsurv_vals = all_data[all_data['Survived'] == 0][key].reset_index(drop = True)
        surv_vals = all_data[all_data['Survived'] == 1][key].reset_index(drop = True)
        plt.hist(nonsurv_vals, bins = bins, alpha = 0.6,
                 color = 'red', label = 'Did not survive')
        plt.hist(surv_vals, bins = bins, alpha = 0.6,
                 color = 'green', label = 'Survived')
    
        # Add legend to plot
        plt.xlim(0, bins.max())
        plt.legend(framealpha = 0.8)
    
    # 'Categorical' features
    else:
       
        # Set the various categories
        if(key == 'Pclass'):
            values = np.arange(1,4)
        if(key == 'Parch' or key == 'SibSp'):
            values = np.arange(0,np.max(data[key]) + 1)
        if(key == 'Embarked'):
            values = ['C', 'Q', 'S']
        if(key == 'Sex'):
            values = ['male', 'female']

        # Create DataFrame containing categories and count of each
        frame = pd.DataFrame(index = np.arange(len(values)), columns=(key,'Survived','NSurvived'))
        for i, value in enumerate(values):
            frame.loc[i] = [value,                    len(all_data[(all_data['Survived'] == 1) & (all_data[key] == value)]),                    len(all_data[(all_data['Survived'] == 0) & (all_data[key] == value)])]

        # Set the width of each bar
        bar_width = 0.4

        # Display each category's survival rates
        for i in np.arange(len(frame)):
            nonsurv_bar = plt.bar(i-bar_width, frame.loc[i]['NSurvived'], width = bar_width, color = 'r')
            surv_bar = plt.bar(i, frame.loc[i]['Survived'], width = bar_width, color = 'g')

            plt.xticks(np.arange(len(frame)), values)
            plt.legend((nonsurv_bar[0], surv_bar[0]),('Did not survive', 'Survived'), framealpha = 0.8)

    # Common attributes for plot formatting
    plt.xlabel(key)
    plt.ylabel('Number of Passengers')
    plt.title('Passenger Survival Statistics With \'%s\' Feature'%(key))
    plt.show()

    # Report number of passengers with missing values
    if sum(pd.isnull(all_data[key])):
        nan_outcomes = all_data[pd.isnull(all_data[key])]['Survived']
        #print "Passengers with missing '{}' values: {} ({} survived, {} did not survive)".format( \
             # key, len(nan_outcomes), sum(nan_outcomes == 1), sum(nan_outcomes == 0))

survival_stats(data, outcomes, 'Sex')


# Let's take a look at whether the feature Sex has any indication of survival rates among passengers using the survival_stats function.
# This function is defined above. The first two parameters passed to the function are the RMS Titanic data and passenger survival outcomes, respectively. The third parameter indicates which feature we want to plot survival statistics across.
# Run the code cell below to plot the survival outcomes of passengers based on their sex.
# 
# # vs.survival_stats(data, outcomes, 'Sex')
# 
# Examining the survival statistics, a large majority of males did not survive the ship sinking. However, a majority of females did survive the ship sinking.
# 
# Let's build on our previous prediction: If a passenger was female, then we will predict that they survived. Otherwise, we will predict the passenger did not survive.
# 
# 

# In[26]:

def predictions_1(data):
    """ Model with one feature: 
            - Predict a passenger survived if they are female. """
    
    predictions = []
    for _, passenger in data.iterrows():
        
        # Remove the 'pass' statement below 
        # and write your prediction conditions here
        pass
    
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_1(data)


# # Question 2
# How accurate would a prediction be that all female passengers survived and the remaining passengers did not survive?

# In[34]:

print (accuracy_score(outcomes, predictions))


# Using just the Sex feature for each passenger, we are able to increase the accuracy of our predictions by a significant margin. Now, let's consider using an additional feature to see if we can further improve our predictions. For example, consider all of the male passengers aboard the RMS Titanic: Can we find a subset of those passengers that had a higher rate of survival? Let's start by looking at the Age of each male, by again using the survival_stats function. This time, we'll use a fourth parameter to filter out the data so that only passengers with the Sex 'male' will be included.
# Run the code cell below to plot the survival outcomes of male passengers based on their age.

# In[35]:

survival_stats(data, outcomes, 'Age', ["Sex == 'male'"])


# Examining the survival statistics, the majority of males younger than 10 survived the ship sinking, whereas most males age 10 or older did not survive the ship sinking. Let's continue to build on our previous prediction: If a passenger was female, then we will predict they survive. If a passenger was male and younger than 10, then we will also predict they survive. Otherwise, we will predict they do not survive.

# In[36]:

def predictions_2(data):
    """ Model with two features: 
            - Predict a passenger survived if they are female.
            - Predict a passenger survived if they are male and younger than 10. """
    
    predictions = []
    for _, passenger in data.iterrows():
        
        # Remove the 'pass' statement below 
        # and write your prediction conditions here
        pass
    
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_2(data)


# # QUESTION 3
# How accurate would a prediction be that all female passengers and all male passengers younger than 10 survived?

# In[37]:

print (accuracy_score(outcomes, predictions))


# Adding the feature Age as a condition in conjunction with Sex improves the accuracy by a small margin more than with simply using the feature Sex alone. Now it's your turn: Find a series of features and conditions to split the data on to obtain an outcome prediction accuracy of at least 80%. This may require multiple features and multiple levels of conditional statements to succeed. You can use the same feature multiple times with different conditions.
# Pclass, Sex, Age, SibSp, and Parch are some suggested features to try.
# Use the survival_stats function below to to examine various survival statistics.
# Hint: To use mulitple filter conditions, put each condition in the list passed as the last argument. Example: ["Sex == 'male'", "Age < 18"]

# In[38]:

survival_stats(data, outcomes, 'Age', ["Sex == 'male'", "Age < 18"])


# After exploring the survival statistics visualization, fill in the missing code below so that the function will make your prediction.
# Make sure to keep track of the various features and conditions you tried before arriving at your final prediction model.

# In[39]:

def predictions_3(data):
    """ Model with multiple features. Makes a prediction with an accuracy of at least 80%. """
    
    predictions = []
    for _, passenger in data.iterrows():
        
        # Removing the 'pass' statement below 
        # and write your prediction conditions here
        pass
    
    # Returning our predictions
    return pd.Series(predictions)

# Making the predictions
predictions = predictions_3(data)


# Describe the steps you took to implement the final prediction model so that it got an accuracy of at least 80%. What features did you look at? Were certain features more informative than others? Which conditions did you use to split the survival outcomes in the data? How accurate are your predictions?

# In[41]:

print (accuracy_score(outcomes, predictions))


# # conclusion
# After several iterations of exploring and conditioning on the data,I have built a useful algorithm for predicting the survival of each passenger aboard the RMS Titanic. The technique applied in this project is a manual implementation of a simple machine learning model, the decision tree. A decision tree splits a set of data into smaller and smaller groups (called nodes), by one feature at a time. Each time a subset of the data is split, our predictions become more accurate if each of the resulting subgroups are more homogeneous (contain similar labels) than before. The advantage of having a computer do things for us is that it will be more exhaustive and more precise than our manual exploration above. This link provides another introduction into machine learning using a decision tree.
# 
# A decision tree is just one of many models that come from supervised learning. In supervised learning, we attempt to use features of the data to predict or model things with objective outcome labels. That is to say, each of our data points has a known outcome value, such as a categorical, discrete label like 'Survived', or a numerical, continuous value like predicting the price of a house.
