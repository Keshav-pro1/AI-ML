import numpy as np

def grad_descent(x, y, theta , learn_rate, iterations):
    m=len(y)
    for _ in range(iterations):
        prediction = np.dot(x,theta)
        errors = prediction - y
        gradient = (1/m) * np.dot(x.T , errors)
        theta = theta - learn_rate * gradient
        
    return theta

a= np.array([[1,2],[3,4]])
b= np.array([5,6])
theta = np.array([0.1,0.2])
learn_rate = 0.01
iterations = 1000
optimized_theta = grad_descent(a,b,theta,learn_rate,iterations)
print("Optimized Theta:\n", optimized_theta)
print