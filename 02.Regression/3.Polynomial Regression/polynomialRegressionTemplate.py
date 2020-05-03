# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

#Fitting the regressor Model
#Create Regressor Here



#predicting using polynomial regression
y_pred=regressor.predict([[6.5]])

#Visualising the regression
plt.scatter(X,y,color="Red")
plt.plot(X,regressor.predict(X),color="Blue")
plt.title("Truth or Bluff(Regression)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

#Visualising the regression(for higher resolution and smoother curve)
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color="Red")
plt.plot(X_grid,regressor.predict(X_grid),color="Blue")
plt.title("Truth or Bluff(Regression)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()


