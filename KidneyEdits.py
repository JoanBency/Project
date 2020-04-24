# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:52:26 2020

@author: JYOTHISH
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 12:13:13 2020

@author: JYOTHISH
"""


import numpy as np
import cv2
import csv
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation 

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as csvfile:
        # Create a writer object from csv module
        csv_writer =  csv.writer(csvfile)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


def delete(filename):
    lines = list()

    members='xx'

    with open(filename, 'r') as readFile:

        reader = csv.reader(readFile)

        for row in reader:

            lines.append(row)
            
        for field in row:
            
            print("Filed values",field)

            if field == members:
                
                print("sucees field",field)
                
                
                
"""
    with open(filename, 'w') as writeFile:

        writer = csv.writer(writeFile)

        writer.writerows(lines)
      
        
 """       

            

def getdata():
    
    #inputs  
    a=input("Enter Age :")
    b=input("Enter Blood Pressure:")
    c=input("Enter Specific Gravity:")
    d=input("Enter Albumin :")
    e=input("Enter Sugar :")
    f=input("Enter Red Blood Cells :")
    g=input("Enter Pus Cell :")
    h=input("Enter Pus Cell clumps :")
    i=input("Enter Bacteria :")
    j=input("Enter Blood Glucose Random :")
    k=input("Enter Blood Urea:")
    l=input("Enter Serum Creatinine :")
    m=input("Enter Sodium :")
    n=input("Enter Potassium :")
    o=input("Enter Hemoglobin :")
    p=input("Enter Packed  Cell Volume :")
    q=input("Enter White Blood Cell Count :")
    r=input("Enter Red Blood Cell Count :")
    s=input("Enter Hypertension :")
    t=input("Enter Diabetes Mellitus :")
    u=input("Enter Coronary Artery Disease :")
    v=input("Enter Appetite :")
    w=input("Enter Pedal Edema :")
    x=input("Enter Anemia:")
    

    
    
    rows = [a,b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x]
    return rows




#####################  CLASSIFIER AND ITS ACCIRACY###########
df = pd.read_csv("D:\\AMAL\\PYTHON\\Preprocessed.csv");


print('ACCURACY OF MODEL')
# TRAINING PROCESS

X = df.values[:, 0:24]
Y = df.values[:, -1]

print("X value",X)

print("Y value",Y)

print(X.shape)
print(Y.shape)

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

clf_gini = DecisionTreeClassifier(criterion = "gini",random_state = 10,max_depth=3, min_samples_leaf=5)

clf_gini.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)
print("Trained output",y_pred)

a= accuracy_score(y_test,y_pred)*100
print("Accuracy",a)

#########################    PREDICTION SYSTEM     ##############################


print("PREDICTION USING MODEL")


filename = "Preprocessed.csv"

r=getdata()

df = pd.read_csv("D:\\AMAL\\PYTHON\\Preprocessed.csv");

#append_list_as_row(filename,r)


#append_list_as_row(filename,r)

L=df.loc[[len(df)-1]]

L = L.values[:, 0:24]

y_pred=clf_gini.predict(L)

print("Trained output",y_pred)

#r.pop(24)

#print("value list", r)

#df = pd.read_csv("D:\\AMAL\\PYTHON\\UserData.csv");

filename = "User.csv"
r.append(y_pred)
print(" new r value",r)

df = pd.read_csv("D:\\AMAL\\PYTHON\\User.csv");
append_list_as_row(filename,r)

#df = df.dropna(axis=0)

#print(df)

#  ENTROPY


clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth = 3, min_samples_leaf = 5)
 
# Performing training
clf_entropy.fit(X_train, y_train)

y_pd = clf_entropy.predict(X_test)


L=df.loc[[len(df)-1]]

L = L.values[:, 0:24]

y_pd = clf_entropy.predict(L)


print("Prediction Using Entropy",y_pd )



logistic_regression = LogisticRegression()

logistic_regression.fit(X_train,y_train)
#y_p=logistic_regression.predict(X_test)


L=df.loc[[len(df)-1]]

L = L.values[:, 0:24]


y_p=logistic_regression.predict(L)
print("Prediction using Logistical regression ",y_p)
