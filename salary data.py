# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 12:12:48 2019

@author: Adarsh
"""

import pandas as pd 
import numpy as np 

# Reading the Salary Data 
salary_train = pd.read_csv("E:/ADM/Excelr solutions/DS assignments/naive bayes/SalaryData_Train.csv")
salary_test = pd.read_csv("E:/ADM/Excelr solutions/DS assignments/naive bayes/SalaryData_Test.csv")

string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]

from sklearn import preprocessing
number = preprocessing.LabelEncoder()
for i in string_columns:
    salary_train[i] = number.fit_transform(salary_train[i])
    salary_test[i] = number.fit_transform(salary_test[i])
	

from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
salary_train["Salary"]=lb.fit_transform(salary_train["Salary"])
salary_test["Salary"]=lb.fit_transform(salary_test["Salary"])	
colnames = salary_train.columns

from sklearn.svm import SVC

train_X = salary_train.iloc[:,0:13]
train_y = salary_train.iloc[:,13]
test_X  = salary_test.iloc[:,0:13]
test_y  = salary_test.iloc[:,13]


# Create SVM classification object 
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'

# kernel = linear
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)

np.mean(pred_test_linear==test_y) # Accuracy = 82.03

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)

np.mean(pred_test_poly==test_y) # Accuracy = 80.42

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y) # Accuracy = 85.44


# kernel = sigmoid
model_sig = SVC(kernel = "sigmoid")
model_sig.fit(train_X,train_y)
pred_test_sig = model_sig.predict(test_X)

np.mean(pred_test_sig==test_y) # Accuracy = 75.43

'''
# Kernel rbf model is giving high accuraccy.

'''