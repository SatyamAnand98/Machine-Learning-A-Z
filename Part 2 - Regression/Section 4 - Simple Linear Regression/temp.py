import numpy as nu
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(X_train,Y_train)

y_pred = clf.predict(X_test)
print(clf.score(X_test,Y_test))

plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, clf.predict(X_train), color = 'blue')
plt.title('Salary Vs Experience(Training Set)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()