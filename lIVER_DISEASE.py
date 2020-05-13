# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 20:29:21 2020

@author: JYOTHISH
"""

import numpy as np
import csv
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix



def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as csvfile:
        # Create a writer object from csv module
        csv_writer =  csv.writer(csvfile)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


def update(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as csvfile:
        # Create a writer object from csv module
        csv_writer =  csv.writer(csvfile)
        # Add contents of list as last row in the csv file
        
        
        csv_writer.writerow(list_of_elem)
        
        
def getdata():
    
    #inputs  
    a=input("Enter Age :")
    b=input("Enter Gender: [MALE: 1    FEMALE: 2]")
    c=input("Enter Total_Bilirubin:")
    d=input("Enter Direct_Bilirubin :")
    e=input("Enter Alkaline_Phosphotase:")
    f=input("Enter Alamine_Aminotransferase :")
    g=input("Enter Aspartate_Aminotransferase :")
    h=input("Enter Total_Protiens :")
    i=input("Enter Albumin :")
    j=input("Enter Albumin_and_Globulin_Ratio:")
    
    
    rows = [a,b, c, d, e, f, g, h, i, j]
    return rows


############################################################################
df = pd.read_csv("D:\\AMAL\\PYTHON\\L.csv");
df.dropna()

filename='L.csv'



#update(filename,rows)
print('ACCURACY OF MODEL')
# TRAINING PROCESS

X = df.values[:, 0:10]
Y = df.values[:, -1]

print("X value",X)

print("Y value",Y)

print(X.shape)
print(Y.shape)

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

print(X_train)
print(X_test)
print(y_train)
print(y_test)



clf_gini = DecisionTreeClassifier(criterion = "entropy",random_state = 10,max_depth=3, min_samples_leaf=5)

print(clf_gini)

print(clf_gini.fit(X_train, y_train))

y_pred = clf_gini.predict(X_test)

#ypp=y_pred
print("Trained output",y_pred)

a= accuracy_score(y_test,y_pred)*100
print("Accuracy",a)

logistic_regression = LogisticRegression()

logistic_regression.fit(X_train,y_train)

y_p=logistic_regression.predict(X_test)
a= accuracy_score(y_test,y_pred)*100
print("Accuracy_LR",a)

###############################################################

print("LIVER DISEASE PREDICTION USING MODEL")


filename = "Liv_User.csv"

r=getdata()

df = pd.read_csv("D:\\AMAL\\PYTHON\\Liv_User.csv");

append_list_as_row(filename,r)

df = pd.read_csv("D:\\AMAL\\PYTHON\\Liv_User.csv");


L=df.loc[[len(df)-1]]

L = L.values[:, 0:10]

print("ENTERED VALUE :  ", L)

y_pred=clf_gini.predict(L)

print("PREDICTED RESULT:  ")
 
print("Trained output",y_pred)

predicted=int(y_pred)

print("Trained output",predicted)


df = pd.read_csv("D:\\AMAL\\PYTHON\\Liv_User.csv");

L=df.loc[[len(df)-1]]
L = L.values[:, 0:10]
y_p=logistic_regression.predict(L)
predicted=int(y_p) 
print("Prediction using Logistical regression ",predicted)

r.append(y_pred)

append_list_as_row(filename,r)


if predicted == 1:
    print("Person Have Liver Disease ")
elif predicted == 2:
    print("Person Dont Have Liver Disease: ")


