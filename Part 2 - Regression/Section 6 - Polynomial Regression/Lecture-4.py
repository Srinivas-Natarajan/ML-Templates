# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#Fitting Linear Regression(Test)
from sklearn.linear_model import LinearRegression
regressor_l = LinearRegression()
regressor_l.fit(X,y)


#Fitting Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
regressor_p = PolynomialFeatures(degree = 2)
X_poly = regressor_p.fit_transform(X)       #Getting polynomial terms
regessor_l_2 = LinearRegression()
regessor_l_2.fit(X_poly,y)

#Visualizing Linear Regression
plt.scatter(X,y,color='red')
plt.plot(X,regressor_l.predict(X),color='blue')
plt.title("Linear Regression Model")
plt.xlabel("Levels")
plt.ylabel("Salary")
plt.show()


#Visualizing Polynomial Regression
plt.scatter(X,y,color='red')
plt.plot(X,regessor_l_2.predict(regressor_p.fit_transform(X)),color='green')  #Predicting with Polynomial Regressor
plt.title("Polynomial Regression Model")                                    #Using Polynomial terms as input
plt.xlabel("Levels")
plt.ylabel("Salary")
plt.show()


#Results of Linear Regression
regressor_l.predict([[6.5]])


#Results of Polynomial Regression
regessor_l_2.predict(regressor_p.fit_transform([[6.5]]))