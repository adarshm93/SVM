# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 17:05:43 2019

@author: Adarsh
"""

import pandas as pd 
import numpy as np 
import seaborn as sns

fire = pd.read_csv("E:/ADM/Excelr solutions/DS assignments/SVM/forestfires (1).csv")
fire.head()
fire.describe()
fire.columns

fire.drop(["month","day"],axis=1,inplace=True) # Dropping the uncessary column 

sns.pairplot(data=fire)

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
train,test = train_test_split(fire,test_size = 0.3)
test.head()
train_X = train.iloc[:,0:28]
train_y = train.iloc[:,28]
test_X  = test.iloc[:,0:28]
test_y  = test.iloc[:,28]

# Create SVM classification object 
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'

# kernel = linear
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)

np.mean(pred_test_linear==test_y) # Accuracy = 98.71

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)

np.mean(pred_test_poly==test_y) # Accuracy = 94.07

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y) # Accuracy = 75.00


# kernel = sigmoid
model_sig = SVC(kernel = "sigmoid")
model_sig.fit(train_X,train_y)
pred_test_sig = model_sig.predict(test_X)

np.mean(pred_test_sig==test_y) # Accuracy = 73.71

'''
#Kernal linear is giving high accurecy= 98.71%

