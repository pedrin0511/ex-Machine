import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime
import ta  


ticker = 'ZS=F' 
start_date = '2023-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')

dados_soja = yf.download(ticker, start=start_date, end=end_date)
preco_atual = dados_soja['Close'].iloc[-1]

print(dados_soja.head())


dados_soja['SMA30'] = ta.trend.sma_indicator(dados_soja['Close'], window=30)
dados_soja['EMA30'] = ta.trend.ema_indicator(dados_soja['Close'], window=30)


dados_soja = dados_soja.reset_index()
dados_soja['Date_ordinal'] = pd.to_datetime(dados_soja['Date']).map(pd.Timestamp.toordinal)

X = dados_soja['Date_ordinal'].values.reshape(-1, 1)  
y = dados_soja['Close'].values 

modelo = LinearRegression()
modelo.fit(X, y)


dados_soja['Trend'] = modelo.predict(X)

inclinacao = modelo.coef_[0]
if inclinacao > 0:
    tendencia = 'Alta'
elif inclinacao < 0:
    tendencia = 'Baixa'
else:
    tendencia = 'Estável'

print(f'Tendência do Mercado: {tendencia}')
print(f'Inclinação da Regressão Linear: {inclinacao:.2f}')

dados_soja['RSI'] = ta.momentum.rsi(dados_soja['Close'], window=14)

if dados_soja['RSI'].iloc[-1] > 70:
    rsi_interpretacao = 'Sobrecomprado - Possível Reversão de Tendência para Baixa'
elif dados_soja['RSI'].iloc[-1] < 30:
    rsi_interpretacao = 'Sobrevendido - Possível Reversão de Tendência para Alta'
else:
    rsi_interpretacao = 'Tendência Atual Mantida'

print(f'Interpretação do RSI: {rsi_interpretacao}')

def gerar_recomendacao(tendencia, rsi):
    if tendencia == 'Alta' and 'Sobrecomprado' not in rsi:
        return 'Vender agora'
    elif tendencia == 'Baixa' or 'Sobrecomprado' in rsi:
        return 'Vender antes que os preços caiam mais'
    elif tendencia == 'Alta' and 'Sobrevendido' in rsi:
        return 'Esperar para vender'
    else:
        return 'Monitorar o mercado'

recomendacao = gerar_recomendacao(tendencia, rsi_interpretacao)
print(f'O valor atual da soja é: ${preco_atual:.2f} por bushel')
print(f'Recomendação: {recomendacao}')

plt.figure(figsize=(14,7))
plt.plot(dados_soja['Date'], dados_soja['Close'], label='Preço de Fechamento', color='blue')
plt.plot(dados_soja['Date'], dados_soja['SMA30'], label='SMA 30 dias', color='orange')
plt.plot(dados_soja['Date'], dados_soja['EMA30'], label='EMA 30 dias', color='green')
plt.plot(dados_soja['Date'], dados_soja['Trend'], label='Linha de Tendência', color='red')
plt.title('Tendência do Preço da Soja com Médias Móveis e Regressão Linear')
plt.xlabel('Data')
plt.ylabel('Preço (USD)')
plt.legend()
plt.show()


plt.figure(figsize=(14,4))
plt.plot(dados_soja['Date'], dados_soja['RSI'], label='RSI 14 dias', color='purple')
plt.axhline(70, color='red', linestyle='--', label='Sobrecomprado (70)')
plt.axhline(30, color='green', linestyle='--', label='Sobrevendido (30)')
plt.title('Índice de Força Relativa (RSI) da Soja')
plt.xlabel('Data')
plt.ylabel('RSI')
plt.legend()
plt.show()
