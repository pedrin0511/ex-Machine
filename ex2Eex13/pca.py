import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.datasets import load_iris


iris = load_iris()
data = iris.data
labels = iris.target


pca = PCA(n_components=2)
data_reduced = pca.fit_transform(data)


explained_variance = pca.explained_variance_ratio_


print(f"Variância explicada pela Componente Principal 1: {explained_variance[0]*100:.2f}%")
print(f"Variância explicada pela Componente Principal 2: {explained_variance[1]*100:.2f}%")


plt.figure(figsize=(8, 6))
plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c=labels, cmap='viridis', marker='o')
plt.title("Dados Iris após redução de dimensionalidade (PCA para 2D)")
plt.xlabel(f"Componente Principal 1 ({explained_variance[0]*100:.2f}% de variância explicada)")
plt.ylabel(f"Componente Principal 2 ({explained_variance[1]*100:.2f}% de variância explicada)")
plt.colorbar(label='Classes')
plt.grid(True)
plt.show()
