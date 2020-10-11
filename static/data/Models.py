# Before building apps we have to have our own models ready for that
# we are training our model and creating it.

# EDA Packages or data loading packages
import pandas as pd
import numpy as np

# ML Packages
from sklearn.feature_extraction.text import CountVectorizer
# further ML Packages will be loaded later

data = pd.read_csv(r"Z:\Data Science data\PrOjects\Python\Name classifier\NationalNames.csv")

# check the head of the data
data.head()

# we only require the name and the gender columns as other not useful
data.columns
data = data[['Name','Gender']]

# lets check whether it is correctly done or not.
data.head()

# let check the size of the dependent variables
data['Gender'].value_counts()

# let's map it as we know that the we require numeric data to perform any sklearn algorithms
data.Gender.replace({'F':0,'M':1},inplace = True)

# again check whether changes has been made or not
data.head()
data.Gender.unique()

# Converting the name into numeric to build models
cv= CountVectorizer()
x = cv.fit_transform(data['Name'])
y = data['Gender']

# storing the cv in file for future use
import joblib
count_vector = open(r'Z:\Data Science data\PrOjects\Python\Name classifier\count_vectoriser.pkl','wb')
joblib.dump(cv,count_vector)
count_vector.close()
# creating train test split to build models
from sklearn.model_selection import train_test_split
x_train, x_test , y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# building Logistic Regression
from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression()
model_lr.fit(x_train,y_train)
print("Accuracy of the training: ",model_lr.score(x_train,y_train))
print("Accuracy of testing data",model_lr.score(x_test,y_test))

import joblib
import os
log_model = open('Z:/Data Science data/PrOjects/Python/Name classifier/log_model_predict.pkl','wb')
joblib.dump(model_lr,log_model)
log_model.close()

# Naive bayes
from sklearn.naive_bayes import MultinomialNB
model_nb= MultinomialNB()
model_nb.fit(x_train,y_train)

print("Accuracy of training : ",model_nb.score(x_train,y_train))
print("Accuracy of testing : ",model_nb.score(x_test,y_test))

# storing it the file
naive_bayes_model = open(r'Z:\Data Science data\PrOjects\Python\Name classifier\nb_model_predict.pkl',"wb")
joblib.dump(model_nb,naive_bayes_model)
naive_bayes_model.close()


# Generating a function to show how the model works and how to interpret it
def genderpredictor(name):
    name = name.capitalize() # just to keep first letter capital
    new_name = [name] # converting it into list for simplification
    vector = cv.transform(new_name).toarray() # converting it into which will the input of the model
    if model_nb.predict(vector) == 0:
        print("FEMALE")
    else:
        print("MALE")

genderpredictor("sophie")


from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier()
model_rf.fit(x_train,y_train)

print("Accuracy of training :",model_rf.score(x_train,y_train))
print("Accuracy of testing :",model_rf.score(x_test,y_test))

# storing it in the system
random_forest = open(r"Z:\Data Science data\PrOjects\Python\Name classifier\rf_model_predict.pkl","wb")
joblib.dump(model_rf,random_forest)
random_forest.close()


