#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset=pd.read_csv("Salary_Data.csv")

#seperating independent variables from dependent

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#dividing the dataset into train and test set

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

#Feature Scaling
'''from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()   #object is created
#Now we need to fit and transform data
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)'''

#Creating a regressor object that will train the model based on taining data
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#Now we will predict using test sets
y_pred=regressor.predict(X_test)

#Visualise the training set using graph
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Salary Vs Experience train")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#Visualise the test set using graph
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Salary Vs Experience test")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()