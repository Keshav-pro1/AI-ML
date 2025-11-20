import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
import matplotlib.pyplot as plt
import math
import random

#Generating dataset
np.random.seed(42)
x =np.random.random((1000,1))*100
y = 2**np.random.randn(1000,1) + 3*x + 5

#split dataset
X_train , X_test,Y_train ,Y_test= train_test_split(x,y,test_size=0.2, random_state=42)

#Fit the Linear Regression model
model = LinearRegression()
model.fit(X_train , Y_train)

#Make predictions
y_pred = model.predict(X_test)

#Evaluate the model
slope = model.coef_[0][0]
intercept = model.intercept_[0]

print("Slope: ", slope)
print("Intercept: ", intercept)

#Check for error
mse = mean_squared_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)
print("Mean Squared Error: ", mse)
print("R^2 Score: ", r2)

# Plotting the result
plt.scatter(X_test, Y_test, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression Result')
plt.legend()
plt.show()

