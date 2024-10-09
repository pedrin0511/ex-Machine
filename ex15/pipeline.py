import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


iris = load_iris()
X = iris.data
y = iris.target


pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('pca', PCA(n_components=2)), 
    ('knn', KNeighborsClassifier(n_neighbors=5)) 
])


scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

print(f"Acurácias por fold: {scores}")
print(f"Acurácia média: {scores.mean():.2f}")
print(f"Desvio padrão da acurácia: {scores.std():.2f}")
