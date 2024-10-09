
import optuna
from optuna.samplers import TPESampler
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings


warnings.filterwarnings('ignore')


iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Conjunto de Treino: {X_train.shape[0]} amostras")
print(f"Conjunto de Teste: {X_test.shape[0]} amostras")


def objective(trial):
   
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 14)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 14)
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])  
    
    #
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        bootstrap=bootstrap,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )

    try:
       
        score = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy').mean()
    except Exception as e:
        
        return 0.0

    return score


sampler = TPESampler(seed=42)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=50, timeout=600) 

print("Melhores Hiperparâmetros Encontrados:")
print(study.best_params)
print(f"Melhor Acurácia CV: {study.best_value * 100:.2f}%\n")


best_params = study.best_params

best_rf = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    bootstrap=best_params['bootstrap'],
    max_features=best_params['max_features'],
    random_state=42,
    n_jobs=-1
)

best_rf.fit(X_train, y_train)


y_pred_best_rf = best_rf.predict(X_test)
accuracy_best_rf = accuracy_score(y_test, y_pred_best_rf)
print(f"Acurácia do Random Forest Otimizado: {accuracy_best_rf * 100:.2f}%\n")

print("Relatório de Classificação do Modelo Otimizado:")
print(classification_report(y_test, y_pred_best_rf, target_names=target_names))

conf_matrix = confusion_matrix(y_test, y_pred_best_rf)
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Previsões')
plt.ylabel('Verdadeiros')
plt.title('Matriz de Confusão - Random Forest Otimizado')
plt.show()


default_rf = RandomForestClassifier(random_state=42, n_jobs=-1)
default_rf.fit(X_train, y_train)

y_pred_default_rf = default_rf.predict(X_test)
accuracy_default_rf = accuracy_score(y_test, y_pred_default_rf)
print(f"Acurácia do Random Forest Padrão: {accuracy_default_rf * 100:.2f}%\n")

print("Relatório de Classificação do Modelo Padrão:")
print(classification_report(y_test, y_pred_default_rf, target_names=target_names))

conf_matrix_default = confusion_matrix(y_test, y_pred_default_rf)
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix_default, annot=True, fmt='d', cmap='Greens', 
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Previsões')
plt.ylabel('Verdadeiros')
plt.title('Matriz de Confusão - Random Forest Padrão')
plt.show()


model_names = ['Random Forest Padrão', 'Random Forest Otimizado']
accuracies = [accuracy_default_rf, accuracy_best_rf]

plt.figure(figsize=(8,6))
sns.barplot(x=model_names, y=accuracies, palette='viridis')
plt.ylabel('Acurácia')
plt.ylim(0.90, 1.00)
for index, value in enumerate(accuracies):
    plt.text(index, value + 0.005, f"{value*100:.2f}%", ha='center', fontsize=12)
plt.title('Comparação de Acurácia entre Modelos')
plt.show()
