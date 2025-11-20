import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generating polynomial dataset
np.random.seed(42)
x = np.random.rand(1000, 1) * 10
y = 2*np.random.randn(1000, 1) + 3*x + x**2 + 5

# Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)

# Split ORIGINAL x
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Transform after splitting
X_train = poly.fit_transform(x_train)
X_test = poly.transform(x_test)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# Correct plotting
plt.scatter(x, y, color='blue', label='Actual Data')
plt.scatter(x_test, y_pred, color='red', label='Polynomial Regression Predictions')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polynomial Regression Result')
plt.legend()
plt.show()
