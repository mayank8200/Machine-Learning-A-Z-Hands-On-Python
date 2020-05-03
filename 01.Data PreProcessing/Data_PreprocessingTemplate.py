#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset=pd.read_csv("Data.csv")

#seperating independent variables from dependent

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#dividing the dataset into train and test set

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

#Feature Scaling
'''from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()   #object is created
#Now we need to fit and transform data
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)'''




