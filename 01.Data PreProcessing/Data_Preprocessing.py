#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset=pd.read_csv("Data.csv")

#seperating independent variables from dependent

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#Handling the missing data
#Imputer takes care o missing data in dataset

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN",strategy="mean",axis=0)
#To select columns from where we have to calculate data
imputer=imputer.fit(X[:,1:3])
#To make changes to the original dataset
X[:,1:3]=imputer.transform(X[:,1:3])

#To deal with categorical data we had to convert it into numbers
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
#converting first column into integer values
X[:,0]=labelencoder_X.fit_transform(X[:,0])
#Encoding categorical data using one hot encoding
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
#Encoding the dependent variables using label encoder
labelencoder_y=LabelEncoder()
#converting first column into integer values
y=labelencoder_y.fit_transform(y)

#dividing the dataset into train and test set

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()   #object is created
#Now we need to fit and transform data
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)




