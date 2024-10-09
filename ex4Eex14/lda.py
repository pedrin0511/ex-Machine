import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=42, stratify=y)


lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)


plt.figure(figsize=(8,6))
colors = ['red', 'green', 'blue']
for label, color, target_name in zip([0,1,2], colors, target_names):
    plt.scatter(X_train_lda[y_train == label, 0], X_train_lda[y_train == label, 1], 
                c=color, label=target_name, edgecolor='k', s=50)
plt.xlabel('Componente LDA 1')
plt.ylabel('Componente LDA 2')
plt.title('Projeção LDA do Conjunto de Dados Iris (Treino)')
plt.legend(loc='best')
plt.grid(True)
plt.show()


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_lda, y_train)


y_pred = knn.predict(X_test_lda)


accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do KNN com LDA: {accuracy*100:.2f}%\n")

print("Relatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=target_names))

print("Matriz de Confusão:")
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Previsões')
plt.ylabel('Verdadeiros')
plt.title('Matriz de Confusão - KNN com LDA')
plt.show()

