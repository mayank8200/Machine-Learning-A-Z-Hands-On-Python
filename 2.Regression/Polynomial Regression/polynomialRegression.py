# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

# Performing Linear Regresion
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

#Performing Polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
poly_reg.fit(X_poly,y)
lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)

#Visualising the linear regression
plt.scatter(X,y,color="Red")
plt.plot(X,lin_reg.predict(X),color="Blue")
plt.title("Truth or Bluff(Linear Regression)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()


#Visualising the Polynomial regression
plt.scatter(X,y,color="Red")
plt.plot(X,lin_reg2.predict(X_poly),color="Blue")
plt.title("Truth or Bluff(Polynomial Regression)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

#predicting using linear regression
lin_reg.predict([[6.5]])

#predicting using polynomial regression
lin_reg2.predict([[1,6.5,6.5**2,6.5**3,6.5**4]])

