import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.datasets import load_iris
iris = load_iris()
data = iris.data

print(iris)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='blue', marker='o')
ax.set_title("Dados originais em 3D")
plt.show()


data_meaned = data - np.mean(data, axis=0)


cov_matrix = np.cov(data_meaned, rowvar=False)


eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)


sorted_idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_idx]
eigenvectors = eigenvectors[:, sorted_idx]


eigenvector_subset = eigenvectors[:, 0:2]


data_reduced = np.dot(data_meaned, eigenvector_subset)


plt.figure(figsize=(8, 6))
plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c='red', marker='o')
plt.title("Dados após redução de dimensionalidade (PCA para 2D)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.grid(True)
plt.show()
