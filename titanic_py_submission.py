# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 14:37:28 2016

Try different models for Titanic Kaggle (Prework for Datascience Course)
 
@author: Annina 

"""

#import packages
#numpy: for maths and arrays
#pandas: for dataframes
#to call sth from these: use eg or np.function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as P
import seaborn as sns
import re
from sklearn import linear_model #for logistic regression
from sklearn.linear_model import LogisticRegression #for logistic regression
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier

########################################################################
#Read in Data
########################################################################
train=pd.read_csv("C:/Users/C930105/Desktop/TitanicKaggle/train.csv",header=0)
test=pd.read_csv("C:/Users/C930105/Desktop/TitanicKaggle/test.csv",header=0)

########################################################################
#Explore Data
########################################################################
train.info() #note that age, cabin and embarked have nulls
test.info() #additionally: Fare is null for 1 entry!

#histograms
train["Age"].dropna().hist(bins=16,range=(0,80),alpha=0.5)
test["Age"].dropna().hist(bins=16,range=(0,80),alpha=0.5) #note: different distribution in class below 20
P.show()

########################################################################
#Clean Data
########################################################################
#keep original train data and store cleaned version in new dataframe
train_clean = train
#object variables should be transformed to numeric
#Why is also train data set transformed????????
train_clean["Sex"] = train_clean["Sex"].map( {"female": 0, "male": 1}).astype(int)
#Wertebereich von Embarked:
train["Embarked"].astype("category")
for i in ['S','C','Q']:
    print (i, len(train[train['Embarked'] == i]))
train_clean['Embarked'] = train['Embarked'].fillna('S') #Syntax to fill missings with S
train_clean["Embarked"] = train['Embarked'].map( {"S": 1, "C": 2, "Q": 3}).astype(int)

#Missing age: replace with median age for the passenger class and gender
#first check, that median age varies with passenger class and gender and 
#that there are enough observations for each passenger class x gender cell
#import train once more, cause sex is also numeric now (???)
train=pd.read_csv("C:/Users/C930105/Desktop/TitanicKaggle/train.csv",header=0)

medians=[]
counts=[]
for i in ["male","female"]:
    for j in range(0,3):
        anz=train[(train.Sex==i) & (train.Pclass==j+1)][["Age"]].count().values[0]
        median=train[(train.Sex==i) & (train.Pclass==j+1)][["Age"]].median().values[0]
        counts.append(anz)
        medians.append(median)
print counts #enough counts for calculating a median
print medians #medians vary, hence replacement is meaningful
        
median_ages = np.zeros((2,3))
for i in range(0,2):
    for j in range(0, 3):
        median_ages[i,j] = train_clean[(train_clean['Sex'] == i) & \
                              (train_clean['Pclass'] == j+1)]['Age'].dropna().median()
median_ages
for i in range(0,2):
    for j in range(0,3):
        train_clean.loc[ (train_clean.Age.isnull()) & (train_clean.Sex == i) & (train_clean.Pclass == j+1), \
                    "Age"] = median_ages[i,j]

train_clean.info()

########################################################################
#Feature Engineering
########################################################################
train_clean["FamilySize"] = train_clean.SibSp + train_clean.Parch + 1

#extract title from name
#train_clean["Name"][1].split()[1] #extracts 2nd word of a string
splitf = lambda x: x.split()[1] #define lambda function to extract 2nd word of a string
train_clean["Title"]=train_clean["Name"].apply(splitf) #applies title extraction function to every row
#note: Title not always 2nd word. 
#new function to extract part between , and . instead of 2nd word
def title_function(name):
    beg_title=name.find(",")
    end_title=name.find(".")
    return name[beg_title+2:end_title]
title = lambda x: title_function(x)
train_clean["Title"]=train_clean["Name"].apply(title)

train_clean.Title.describe() #17 levels
#count which levels are frequent and rare
train_clean.Title.value_counts()

#combine Mlle and Ms with Miss (Fräulein) and Mme with Mrs (Frau)
train_clean.loc[ (train_clean.Title=="Mlle") , "Title"]= "Miss"
train_clean.loc[ (train_clean.Title=="Ms") , "Title"]= "Miss"
train_clean.loc[ (train_clean.Title=="Mme") , "Title"]= "Mrs"
#rich women titles map to "Lady"
train_clean.loc[ (train_clean.Title=="Dona") , "Title"]= "Lady"
train_clean.loc[ (train_clean.Title=="the Countess") , "Title"]= "Lady"
#rich men titles mapping to Sir
train_clean.loc[ (train_clean.Title=="Capt") , "Title"]= "Sir"
train_clean.loc[ (train_clean.Title=="Don") , "Title"]= "Sir"
train_clean.loc[ (train_clean.Title=="Major") , "Title"]= "Sir"
train_clean.loc[ (train_clean.Title=="Jonkheer") , "Title"]= "Sir"

#transform to numeric
train_clean["Title"] = train_clean["Title"].map( {"Mr": 1, "Miss": 2, "Mrs": 3,
  "Master": 4, "Dr": 5, "Rev": 6, "Sir": 7, "Col": 8, "Lady": 9 }).astype(int)

#cabin deck
deck = lambda x: x[0]
dummy=train_clean.Cabin.fillna("Z")
train_clean["Deck"]=dummy.apply(deck)
train_clean.Deck.value_counts()
#check which decks are for which classes
first=train_clean[train_clean.Pclass==1]
second=train_clean[train_clean.Pclass==2]
third=train_clean[train_clean.Pclass==3]
first.Deck.value_counts() #mostly C and B, no F, G
second.Deck.value_counts() #mostly Z, and some F, D, E
third.Deck.value_counts() #mostly Z (ie unknown), and some F, G, E
#Deck T only once, map to A (apparently boat deck and A is right below)
train_clean.Deck.fillna("Z")
train_clean["Deck"] = train_clean["Deck"].map( {"A": 1, "B": 2, "C": 3,
  "D": 4, "E": 5, "F": 6, "G": 7, "T": 1, "Z": 8 }).astype(int)
train_clean.head()

#cabin number (possibly tells sth more about location)
dummy=train_clean.Cabin.fillna("Z999")
cabnr = lambda x: filter(str.isdigit, x)
train_clean["CabinNr"]=dummy.apply(cabnr)
#transform to integer
train_clean.CabinNr[train_clean.CabinNr==""]="999"
train_clean.CabinNr=train_clean.CabinNr.astype(int)
train_clean.info()

train_final=train_clean.drop(["Name","Ticket","Cabin"],axis=1)
########################################################################
#Logistic Regression
########################################################################
logisticr = linear_model.LogisticRegression()
X=train_final.drop(["PassengerId","Survived"],axis=1)
y=train_final["Survived"]
#total training data split in train&test to have test sample with responses
#in order to check performance before submitting

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=42)
pred_var = ['Pclass','Sex','Age','FamilySize','Fare','Embarked',
             'Title','Deck','CabinNr'] #19.0%
logistic_fit=logisticr.fit(X_train[pred_var],y_train)
logisticr.score(X_train[pred_var],y_train)
y_pred = logisticr.predict(X_train[pred_var])
perc_correct=1-sum(abs(y_pred-y_train))/float(len(y_train)) #is about the same as the score

y_pred = logisticr.predict(X_test[pred_var])
error=sum((y_pred-y_test)**2)**0.5
#percentage of correctly classified:
perc_correct=1-sum(abs(y_pred-y_test))/float(len(y_test)) #1-19.7%
logisticr.score(X_test[pred_var],y_test)

#try different predictors
pred_var = ['Pclass','Sex','Age'] #19.0%
pred_var = ['Sex','Age'] #19.0%
pred_var = ['Sex'] #20.3%
pred_var = ['Sex','Deck'] #20.3%
pred_var = ['Pclass','Sex','Deck','Age'] #19.0%
pred_var = ['Pclass','Sex','FamilySize','Deck'] #19.3%
pred_var = ['Pclass','Sex','FamilySize','Fare','Embarked','Title','Deck','CabinNr'] #21.0%
pred_var = ['Pclass','Sex','SibSp','Fare','Embarked','Title','Deck','CabinNr'] #21.4%
pred_var = ['Pclass','Sex','SibSp','Parch','Fare','Embarked','Title','Deck','CabinNr'] #21.0%
pred_var = ['Pclass','Sex','SibSp','FamilySize','Fare','Embarked','Title','Deck','CabinNr'] #21.0%
pred_var = ['Pclass','Sex','Age','FamilySize','SibSp'] #81%

logistic_fit=logisticr.fit(X_train[pred_var],y_train)
score_train=logisticr.score(X_train[pred_var],y_train)
print score_train
y_pred = logisticr.predict(X_test[pred_var])
error=sum((y_pred-y_test)**2)**0.5
#percentage of correctly classified:
perc_correct=1-sum(abs(y_pred-y_test))/float(len(y_test)) 
score_test=logisticr.score(X_test[pred_var],y_test)
print perc_correct
print score_test

scores = cross_validation.cross_val_score(logisticr,X[pred_var],y,cv=10)
print scores
print scores.mean()

#draw ROC curve
logisticr.fit(X_train[pred_var],y_train)
preds = logisticr.predict_proba(X_test[pred_var])[:,1]
fpr, tpr, _ = metrics.roc_curve(y_test, preds)
t=[0,1]
plt.plot(t,t,"r--",fpr,tpr) 

########################################################################
#Decision Tree classifier
########################################################################
baum = DecisionTreeClassifier(random_state=1)
pred_var = ['Pclass','Sex','Age','FamilySize','Fare','Embarked','Title','Deck','CabinNr'] #0.78
#pred_var = ['Pclass','Sex','Age','Title','Deck'] #0.80

baum = baum.fit(X_train[pred_var],y_train)
baum.feature_importances_
output = baum.predict(X_test[pred_var])
baum.score(X_test[pred_var],y_test)
scores = cross_validation.cross_val_score(baum,X[pred_var],y,cv = 10)
print scores.mean()

#try different tuning parameters
baum = DecisionTreeClassifier(random_state=1,min_samples_leaf=4) #0.794
baum = DecisionTreeClassifier(random_state=1,max_depth=4) #0.818
pred_var = ['Pclass','Sex','Age','FamilySize','Fare','Embarked','Title','Deck','CabinNr'] #0.78

baum = baum.fit(X_train[pred_var],y_train)
baum.feature_importances_
baum.tree_
output = baum.predict(X_test[pred_var])
baum.score(X_test[pred_var],y_test)
scores = cross_validation.cross_val_score(baum,X[pred_var],y,cv = 10)
print scores.mean()

#plot the tree
from sklearn import tree
baum2 = tree.DecisionTreeClassifier()
baum2 = baum2.fit(X[pred_var], y)
dotfile=open('C:/Users/C930105/Desktop/TitanicKaggle/plottree.dot',"w")
tree.export_graphviz(baum2,out_file=dotfile,feature_names=X[pred_var].columns)    
dotfile.close()    
from subprocess import check_call
check_call(['dot','-Tpng','C:/Users/C930105/Desktop/TitanicKaggle/plottree.dot','-o','C:/Users/C930105/Desktop/TitanicKaggle/plottree.png'])


########################################################################
#Random Forests
########################################################################
#Import random forest package
from sklearn.ensemble import RandomForestClassifier

#Create random forest object which will include all the param for the fit
forest = RandomForestClassifier(n_estimators = 100, min_samples_leaf=3) #82% score on test, but best score on cross validation 0.832
forest = RandomForestClassifier(n_estimators = 100, max_depth=4)# best score on test: 0.837 with max_depth=4
#forest = RandomForestClassifier(n_estimators = 100, min_samples_split=6) 

#Fit training data to "Survived" and create the decision trees
pred_var = ['Pclass','Sex','Age','FamilySize','Fare','Embarked','Title','Deck','CabinNr']
forest = forest.fit(X_train[pred_var],y_train)
forest.feature_importances_
forest.score(X_train[pred_var],y_train)

#take the same decision trees and run it on the test data
output = forest.predict(X_test[pred_var])
forest.score(X_test[pred_var],y_test) 
#note: the score is the percentage of false predictions, ie
#1-sum(abs(output-y_test))/float(len(y_test)) = score

#cross validation
forest = forest.fit(X[pred_var],y)
scores = cross_validation.cross_val_score(forest,X[pred_var],y,cv = 5)
print scores
print scores.mean()

########################################################################
#Prepare test data with analogous cleaning and features as train data
########################################################################
#keep original test data and store cleaned version in new dataframe
test_clean = test
test_clean["Sex"] = test_clean["Sex"].map( {"female": 0, "male": 1}).astype(int)
#Missing Embarked to "S"
test_clean['Embarked'] = test['Embarked'].fillna('S') #Syntax to fill missings with S
test_clean["Embarked"] = test['Embarked'].map( {"S": 1, "C": 2, "Q": 3}).astype(int)

#import test once more, cause sex is also numeric now (???)
test=pd.read_csv("C:/Users/C930105/Desktop/TitanicKaggle/test.csv",header=0)

medians=[]
counts=[]
for i in ["male","female"]:
    for j in range(0,3):
        anz=test[(test.Sex==i) & (test.Pclass==j+1)][["Age"]].count().values[0]
        median=test[(test.Sex==i) & (test.Pclass==j+1)][["Age"]].median().values[0]
        counts.append(anz)
        medians.append(median)
print counts #enough counts for calculating a median
print medians #medians vary, hence replacement is meaningful
        
median_ages = np.zeros((2,3))
for i in range(0,2):
    for j in range(0, 3):
        median_ages[i,j] = test_clean[(test_clean['Sex'] == i) & \
                              (test_clean['Pclass'] == j+1)]['Age'].dropna().median()
median_ages
for i in range(0,2):
    for j in range(0,3):
        test_clean.loc[ (test_clean.Age.isnull()) & (test_clean.Sex == i) & (test_clean.Pclass == j+1), \
                    "Age"] = median_ages[i,j]

test_clean.info()

########################################################################
#Feature Engineering on test data
########################################################################
test_clean["FamilySize"] = test_clean.SibSp + test_clean.Parch + 1

test_clean["Title"]=test_clean["Name"].apply(title)
#count which levels are frequent and rare
test_clean.Title.value_counts()
#combine Mlle and Ms with Miss (Fräulein) and Mme with Mrs (Frau)
test_clean.loc[ (test_clean.Title=="Mlle") , "Title"]= "Miss"
test_clean.loc[ (test_clean.Title=="Ms") , "Title"]= "Miss"
test_clean.loc[ (test_clean.Title=="Mme") , "Title"]= "Mrs"
#rich women titles map to "Lady"
test_clean.loc[ (test_clean.Title=="Dona") , "Title"]= "Lady"
test_clean.loc[ (test_clean.Title=="the Countess") , "Title"]= "Lady"
#rich men titles mapping to Sir
test_clean.loc[ (test_clean.Title=="Capt") , "Title"]= "Sir"
test_clean.loc[ (test_clean.Title=="Don") , "Title"]= "Sir"
test_clean.loc[ (test_clean.Title=="Major") , "Title"]= "Sir"
test_clean.loc[ (test_clean.Title=="Jonkheer") , "Title"]= "Sir"

#transform to numeric
test_clean["Title"] = test_clean["Title"].map( {"Mr": 1, "Miss": 2, "Mrs": 3,
  "Master": 4, "Dr": 5, "Rev": 6, "Sir": 7, "Col": 8, "Lady": 9 }).astype(int)

#cabin deck
deck = lambda x: x[0]
dummy=test_clean.Cabin.fillna("Z")
test_clean["Deck"]=dummy.apply(deck)
test_clean.Deck.value_counts()
#no Deck T in test data
test_clean.Deck.fillna("Z")
test_clean["Deck"] = test_clean["Deck"].map( {"A": 1, "B": 2, "C": 3,
  "D": 4, "E": 5, "F": 6, "G": 7, "T": 1, "Z": 8 }).astype(int)
test_clean.head()

#cabin number (possibly tells sth more about location)
dummy=test_clean.Cabin.fillna("Z999")
cabnr = lambda x: filter(str.isdigit, x)
test_clean["CabinNr"]=dummy.apply(cabnr)
#transform to integer
test_clean.CabinNr[test_clean.CabinNr==""]="999"
test_clean.CabinNr=test_clean.CabinNr.astype(int)
test_clean.info()
#note: one unknown Fare, 
test_clean["Fare"].dropna().hist(bins=16,range=(0,80),alpha=0.5)
#use mean ticket fare of Pclass
test_clean[test_clean.Fare.isnull()] #thirc class ticket
thirdclass=test_clean[test_clean.Pclass==3]
test_clean.Fare=test_clean.Fare.fillna(thirdclass.Fare.mean())

test_final=test_clean.drop(["Name","Ticket","Cabin"],axis=1)

########################################################################
#Apply models to real test data
########################################################################

########################################################################
#Logistic Regression
########################################################################
logisticr = linear_model.LogisticRegression()
logistic_fit=logisticr.fit(X[pred_var],y)
#try different predictors
pred_var = ['Pclass','Sex','Deck','Age'] #79.8%
#pred_var = ['Pclass','Sex','Age','FamilySize','SibSp'] #79.7%
scores = cross_validation.cross_val_score(logisticr,X[pred_var],y,cv=10)
print scores.mean()
X_test=test_final.drop(["PassengerId"],axis=1)
y_pred = logisticr.predict(X_test[pred_var])
sub_logReg = pd.DataFrame({"PassengerId":test_final["PassengerId"],"Survived":y_pred})
sub_logReg.to_csv("C:/Users/C930105/Desktop/TitanicKaggle/logisticprediciton1.csv",index=False)

########################################################################
#Decision Tree classifier
########################################################################
#try different tuning parameters
baum = DecisionTreeClassifier(random_state=1,max_depth=4) #0.818
pred_var = ['Pclass','Sex','Age','FamilySize','Fare','Embarked','Title','Deck','CabinNr'] #0.78

baum = baum.fit(X[pred_var],y)
scores = cross_validation.cross_val_score(baum,X[pred_var],y,cv = 10)
print scores.mean()
y_pred = baum.predict(X_test[pred_var])

sub_tree = pd.DataFrame({"PassengerId":test_final["PassengerId"],"Survived":y_pred})
sub_tree.to_csv("C:/Users/C930105/Desktop/TitanicKaggle/treeprediciton.csv",index=False)

########################################################################
#Random Forests
########################################################################
#Import random forest package
from sklearn.ensemble import RandomForestClassifier

#Create random forest object which will include all the param for the fit
forest = RandomForestClassifier(n_estimators = 100, min_samples_leaf=3) #82% score on test, but best score on cross validation 0.832

#Fit training data to "Survived" and create the decision trees
pred_var = ['Pclass','Sex','Age','FamilySize','Fare','Embarked','Title','Deck','CabinNr']
forest = forest.fit(X_train[pred_var],y_train)
#cross validation
forest = forest.fit(X[pred_var],y)
scores = cross_validation.cross_val_score(forest,X[pred_var],y,cv = 5)
print scores.mean()
#take the same decision trees and run it on the test data
y_pred = forest.predict(X_test[pred_var])
sub_forest = pd.DataFrame({"PassengerId":test_final["PassengerId"],"Survived":y_pred})
sub_forest.to_csv("C:/Users/C930105/Desktop/TitanicKaggle/forestprediciton.csv",index=False)
