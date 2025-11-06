import numpy as np

arr = np.array([11,22,33,44,55])
print(arr)

zero = np.zeros((3,3))
print(zero)

one = np.ones((4,4,))
print(one)

range_builder = np.arange(1,20,3)
print(range_builder)

linspace_array = np.linspace(0,15,4)
print(linspace_array)

print("\n")

linspace_builder_reshaped = linspace_array.reshape((4,1))
print(linspace_builder_reshaped)

transpose = linspace_builder_reshaped.T
print("Transpose", transpose)