import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')



iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Conjunto de Treino: {X_train.shape[0]} amostras")
print(f"Conjunto de Teste: {X_test.shape[0]} amostras")


rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)


y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Precisão do Random Forest: {accuracy_rf:.4f}")

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]


plt.figure(figsize=(10,6))
plt.title("Importância das Características - Random Forest")
sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette="viridis")
plt.xlabel("Importância")
plt.ylabel("Características")
plt.show()


estimator = rf.estimators_[0]

plt.figure(figsize=(20,10))
plot_tree(estimator, feature_names=feature_names, class_names=target_names, filled=True, rounded=True)
plt.title("Árvore Individual do Random Forest")
plt.show()
