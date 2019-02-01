# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
lg = LinearRegression()
lg.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree = 4)
X_poly = pr.fit_transform(X)
lg2 = LinearRegression()
lg2.fit(X_poly, y)

plt.scatter(X, y, color='red')
plt.plot(X, lg.predict(X), color='blue')
plt.title('Truth Or Bluff(Linear Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

xg = np.arange(min(X), max(X), 0.01)
'''here the plotting goes from 1 to 10 at an interval of 0.01 i.e. against each element 
from 1 to 10 at an interval of 0.01 is plotted and not only against the elements given 
in the dataset'''
xg = xg.reshape((len(xg),1))
plt.scatter(X, y, color='red')
plt.plot(xg, lg2.predict(pr.fit_transform(xg)), color='blue')
plt.title('Truth Or Bluff(Polynomial Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#print(lg.predict(6.5))
#print(lg2.predict(pr.fit_transform(6.5)))
plt.plot(X,y,'-o')