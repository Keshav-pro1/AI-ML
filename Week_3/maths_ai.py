import numpy as np

# Dot product
A = np.array([[1,2],[3,4]])
B = np.array([[9,8],[6,7]])
result = np.dot(A,B)
print(result)

#Identity Matrix
I = np.eye(3)
print(I)

#Diagonal Matrix
D = np.diag([1,2,3])
print(D)

#Matrix Inversion
M = np.array([[4,7],[2,6]])
M_inv = np.linalg.inv(M)
print(M_inv)

# Determenant 
det_M = np.linalg.det(M)
print(round(det_M , 2))

#EigenValues and Eigen Vectors
eigenvalues, eigenvectors = np.linalg.eig(M)
print("Eigenvalues:", eigenvalues)