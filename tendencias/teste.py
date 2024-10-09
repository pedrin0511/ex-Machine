import yfinance as yf
import matplotlib.pyplot as plt


tickers_futuros = ['ZSF25.CBT', 'ZSH25.CBT', 'ZSK25.CBT', 'ZSN25.CBT','ZSQ25.CBT']  # Janeiro, Março e Maio de 2025


meses = {
    'ZSF25.CBT': 'Janeiro',
    'ZSH25.CBT': 'Março',
    'ZSK25.CBT': 'Maio',
    'ZSN25.CBT': 'Julio',
    'ZSQ25.CBT': 'Agosto'
}

precos = {}


for ticker in tickers_futuros:
    soja = yf.Ticker(ticker)
    dados_futuros = soja.history(period='1d')
    if not dados_futuros.empty:
        preco_atual = dados_futuros['Close'].iloc[-1]
        precos[ticker] = preco_atual


nomes_meses = list(meses.values())
valores_precos = [precos[ticker] for ticker in tickers_futuros]


plt.figure(figsize=(10, 6))
plt.plot(nomes_meses, valores_precos, marker='o')
plt.title('Tendência Futura dos Preços do Bushel de Soja')
plt.xlabel('Meses')
plt.ylabel('Preço em Dólares por Bushel')
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

