import numpy as np

#Array and scalar broadcasting
arr = np.array ([1,2,3,4,5,6])
print(arr+10)

matrix= np.array([[1,2,3],[4,5,6],[7,8,9]])
vector = np.array([1,0,1])
print(matrix + vector)
print("\n")


# Aggregation function: statistical operations
data = np.array([[1,2,3],[4,5,6]])
print("Sum: ", np.sum(data))
print("Mean:", np.mean(data))
print("Median", np.median(data))
print("Standard Deviation:", np.std(data))
print("Variance:", np.var(data))
print("Min:", np.min(data))
print("Max:", np.max(data))
print("Sum along row", np.sum(data , axis=1))
print("Sum along column", np.sum(data , axis=0))
print("\n")

#Filtering
raw_data = np.array([5,6,7,1,2,46,89])
evens = raw_data[raw_data % 2 == 0]
print("Even numbers:", evens)

raw_data[raw_data >2] = 9
print("Modified array:", raw_data)
print("\n")

#Random numbers and Seeding
np.random.seed(42)
random_array = np.random.rand(3,3)
print("Random Array:\n", random_array)

random_integers = np.random.randint(1,10,size=(4,2))
print("Random Integers:\n", random_integers)
