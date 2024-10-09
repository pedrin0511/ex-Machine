import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris


iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
colors = ['red', 'green', 'blue']
for label, color, target_name in zip([0,1,2], colors, target_names):
    ax.scatter(X[y == label, 0], X[y == label, 1], X[y == label, 2], 
               c=color, label=target_name, edgecolor='k', s=50)
ax.set_title("Dados Originais Iris em 3D")
ax.set_xlabel(iris.feature_names[0])
ax.set_ylabel(iris.feature_names[1])
ax.set_zlabel(iris.feature_names[2])
ax.legend()
plt.show()


mean_overall = np.mean(X, axis=0)


classes = np.unique(y)
mean_vectors = []
for cls in classes:
    mean_vectors.append(np.mean(X[y == cls], axis=0))


Sw = np.zeros((X.shape[1], X.shape[1]))
for cls, mean_vec in zip(classes, mean_vectors):
    class_scatter = np.zeros((X.shape[1], X.shape[1]))
    for row in X[y == cls]:
        row, mean_vec = row.reshape(X.shape[1],1), mean_vec.reshape(X.shape[1],1)
        class_scatter += (row - mean_vec).dot((row - mean_vec).T)
    Sw += class_scatter


Sb = np.zeros((X.shape[1], X.shape[1]))
for cls, mean_vec in zip(classes, mean_vectors):
    n = X[y == cls].shape[0]
    mean_vec = mean_vec.reshape(X.shape[1],1)
    mean_overall_vec = mean_overall.reshape(X.shape[1],1)
    Sb += n * (mean_vec - mean_overall_vec).dot((mean_vec - mean_overall_vec).T)


eig_vals, eig_vecs = np.linalg.eig(np.linalg.pinv(Sw).dot(Sb))


eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)


W = np.hstack((eig_pairs[0][1].reshape(X.shape[1],1),
               eig_pairs[1][1].reshape(X.shape[1],1)))


X_lda = X.dot(W)


plt.figure(figsize=(8,6))
for label, color, target_name in zip([0,1,2], colors, target_names):
    plt.scatter(X_lda[y == label, 0], X_lda[y == label, 1], 
                c=color, label=target_name, edgecolor='k', s=50)
plt.xlabel('Componente LDA 1')
plt.ylabel('Componente LDA 2')
plt.title('Projeção LDA do Conjunto de Dados Iris')
plt.legend(loc='best')
plt.grid(True)
plt.show()
