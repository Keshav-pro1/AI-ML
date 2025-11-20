import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression ,Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Generating polynomial dataset
np.random.seed(42)
x = np.random.rand(100, 1) * 10
y = 2*np.random.randn(100, 1) + 3*x + x**2 + 5

# Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)

# Split ORIGINAL x
x_train_raw, x_test_raw, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Now apply polynomial transform
x_train = poly.fit_transform(x_train_raw)
x_test = poly.transform(x_test_raw)

# Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(x_train, y_train)
y_ridge_pred = ridge_model.predict(x_test)

# Lasso Regression
lasso_model = Lasso(alpha=0.1)   # important: smaller alpha works better for this data
lasso_model.fit(x_train, y_train)
y_lasso_pred = lasso_model.predict(x_test)

# Evaluation
print("Ridge Regression - MSE:", mean_squared_error(y_test, y_ridge_pred))
print("Ridge Regression - R^2:", r2_score(y_test, y_ridge_pred))

print("Lasso Regression - MSE:", mean_squared_error(y_test, y_lasso_pred))
print("Lasso Regression - R^2:", r2_score(y_test, y_lasso_pred))

# Plotting
plt.scatter(x, y, color='blue', label='Actual Data')
plt.scatter(x_test_raw, y_ridge_pred, color='red', label='Ridge Predictions')
plt.scatter(x_test_raw, y_lasso_pred, color='green', label='Lasso Predictions')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Ridge and Lasso Regression Results')
plt.legend()
plt.show()
