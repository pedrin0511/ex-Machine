import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('./exemplo_produtos.xlsx')

df = df.ffill()
df['infos'] = df['nome'] + '' + df['descricao'] + ' ' + df['categoria'] + '' + df['Marca']

importancia_palavra = TfidfVectorizer(stop_words='english')
importancia_palavra_matrix = importancia_palavra.fit_transform(df['infos'])

similaridade = cosine_similarity(importancia_palavra_matrix, importancia_palavra_matrix)

df = df.reset_index()
indices = pd.Series(df.index, index=df['id']).drop_duplicates()

def recomendacao_produtos(id, similaridade=similaridade):
    if id not in indices:
        print("Produto não encontrado")
        return None
    idx = indices[id]
    scores = list(enumerate(similaridade[idx]))
    scores = sorted(scores, key=lambda x : x[1], reverse=True)
    scores = scores[1:11]
    indices_produto = [i[0] for i in scores]
    return df[['id', 'nome', 'categoria', 'Marca']].iloc[indices_produto]

id = 1
recomendacao = recomendacao_produtos(id)

print(f"Recomendações para o Produto ID {id}:")
print(recomendacao)