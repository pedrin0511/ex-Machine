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


# Carregar o conjunto de dados Iris
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Conjunto de Treino: {X_train.shape[0]} amostras")
print(f"Conjunto de Teste: {X_test.shape[0]} amostras")

class BaggingClassifierCustom:
    def __init__(self, base_estimator, n_estimators=10, max_samples=1.0, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.estimators_ = []
        self.bootstrap_indices_ = []
        
    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        sample_size = int(self.max_samples * n_samples)
        
        for i in range(self.n_estimators):
    
            indices = np.random.choice(n_samples, size=sample_size, replace=True)
            self.bootstrap_indices_.append(indices)
            X_sample, y_sample = X[indices], y[indices]
            
        
            estimator = self.base_estimator()
            estimator.fit(X_sample, y_sample)
            self.estimators_.append(estimator)
        
    def predict(self, X):

        predictions = np.array([estimator.predict(X) for estimator in self.estimators_])
   
        majority_votes = [Counter(row).most_common(1)[0][0] for row in predictions.T]
        return np.array(majority_votes)

bagging_custom = BaggingClassifierCustom(
    base_estimator=lambda: DecisionTreeClassifier(max_depth=None, random_state=42),
    n_estimators=50,
    max_samples=1.0,
    random_state=42
)

bagging_custom.fit(X_train, y_train)
y_pred_bagging = bagging_custom.predict(X_test)


accuracy_bagging = accuracy_score(y_test, y_pred_bagging)
print(f"Precisão do Bagging Custom: {accuracy_bagging:.4f}")


single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train, y_train)
y_pred_tree = single_tree.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print(f"Precisão da Árvore de Decisão Simples: {accuracy_tree:.4f}")


