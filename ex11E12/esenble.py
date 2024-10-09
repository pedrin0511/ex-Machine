# Importação das Bibliotecas Necessárias
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
    RandomForestClassifier
)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')



sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Conjunto de Treino: {X_train.shape[0]} amostras")
print(f"Conjunto de Teste: {X_test.shape[0]} amostras")


# Exercício 11: Métodos Ensemble - Boosting


print("\n--- Exercício 11: Métodos Ensemble - Boosting ---\n")


single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train, y_train)
y_pred_tree = single_tree.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print(f"Precisão da Árvore de Decisão Simples: {accuracy_tree:.4f}")


ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=50,
    random_state=42
)
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)
accuracy_ada = accuracy_score(y_test, y_pred_ada)
print(f"Precisão do AdaBoost: {accuracy_ada:.4f}")


gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print(f"Precisão do Gradient Boosting: {accuracy_gb:.4f}")


plt.figure(figsize=(20,10))
plot_tree(gb.estimators_[0, 0], feature_names=feature_names, class_names=target_names, filled=True, rounded=True)
plt.title("Primeira Árvore do Gradient Boosting")
plt.show()





print("\n--- Exercício 12: Comparação de Métodos Ensemble ---\n")


models = {
    'Árvore de Decisão Simples': DecisionTreeClassifier(random_state=42),
    'Bagging': BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    'AdaBoost': AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
        n_estimators=50,
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
}


results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append({'Modelo': name, 'Precisão': acc})
    print(f"Precisão do {name}: {acc:.4f}")


results_df = pd.DataFrame(results)
print("\nComparação de Precisões dos Modelos Ensemble:")
print(results_df)


for name in ['Random Forest', 'Gradient Boosting']:
    model = models[name]
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10,6))
    plt.title(f"Importância das Características - {name}")
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette="viridis")
    plt.xlabel("Importância")
    plt.ylabel("Características")
    plt.show()


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(10,6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Tamanho do Conjunto de Treino")
    plt.ylabel("Precisão")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='accuracy'
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.grid()
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Treino")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validação")
    plt.legend(loc="best")
    plt.show()


for name, model in models.items():
    plot_learning_curve(model, f"Curva de Aprendizado - {name}", X, y)


print("\n--- Análise Comparativa ---\n")
print(results_df.sort_values(by='Precisão', ascending=False))

rf = models['Random Forest']
plt.figure(figsize=(20,10))
plot_tree(rf.estimators_[0], feature_names=feature_names, class_names=target_names, filled=True, rounded=True)
plt.title("Primeira Árvore do Random Forest")
plt.show()


bagging = models['Bagging']
plt.figure(figsize=(20,10))
plot_tree(bagging.estimators_[0], feature_names=feature_names, class_names=target_names, filled=True, rounded=True)
plt.title("Primeira Árvore do Bagging")
plt.show()
