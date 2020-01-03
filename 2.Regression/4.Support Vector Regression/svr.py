# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()   
sc_y=StandardScaler()        #object is created
#Now we need to fit and transform data
X=sc_X.fit_transform(X)
y=sc_y.fit_transform(y)

#Fitting the regressor Model
#Create Regressor Here
from sklearn.svm import SVR
regressor=SVR(kernel="rbf")
regressor.fit(X,y)


#predicting using SVR regression
y_pred=regressor.predict([[6.5]])

#Visualising the SVR regression
plt.scatter(X,y,color="Red")
plt.plot(X,regressor.predict(X),color="Blue")
plt.title("Truth or Bluff(SVR Regression)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()
