import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Carregar dados do arquivo XLSX
try:
    df = pd.read_excel('dados_estudo.xlsx') 
    horas_estudo = df['Hours Studied'].tolist() 
    notas = df['Performance Index'].tolist() 
except FileNotFoundError:
    print("Erro: O arquivo 'dados_estudo.xlsx' não foi encontrado. Verifique o nome e o caminho do arquivo.")
    exit()
except KeyError as e:
    print(f"Erro: Coluna não encontrada no arquivo Excel: {e}. Verifique os nomes das colunas.")
    exit()

# Exploração de dados
print("Head:\n", df.head(), "\n")
print("Describe:\n", df.describe(), "\n")
print("Info:")
df.info()
print("\n")

print("Value Counts - Horas de Estudo:\n", df['Hours Studied'].value_counts(), "\n")
print("Value Counts - Notas:\n", df['Performance Index'].value_counts(), "\n")

# Garantir mesmo tamanho
# n = len(horas_estudo) # n será definido a partir do DataFrame
n = len(df)

# Cálculo da regressão
media_x = sum(horas_estudo) / n
media_y = sum(notas) / n

numerador_m = sum((horas_estudo[i] - media_x) * (notas[i] - media_y) for i in range(n))
denominador_m = sum((horas_estudo[i] - media_x)**2 for i in range(n))

m = numerador_m / denominador_m
b = media_y - m * media_x

print(f"Coeficiente Angular (m): {m:.2f}")
print(f"Intercepto Linear (b): {b:.2f}")
print(f"Equação: Nota = {m:.2f} * Horas_Estudo + {b:.2f}\n");

# Previsões
# horas_para_prever = 7.5
horas_para_prever = input("\nDigite um valor horas para prever a nota!\n->")
horas_para_prever = float(horas_para_prever)
#Calculo de previsão onde m coeficiente angular * horas_prever + b que e o intercepto linear
nota_prevista = m * horas_para_prever + b
print(f"Para {horas_para_prever} horas de estudo, nota prevista: {nota_prevista:.2f}")

notas_previstas = [m * x + b for x in horas_estudo]

# Métricas calculadas manualmente
# MSE, RMSE, MAE, R²
soma_erro_quadrado = sum((notas[i] - notas_previstas[i])**2 for i in range(n))
mse = soma_erro_quadrado / n
rmse = math.sqrt(mse)

soma_erro_absoluto = sum(abs(notas[i] - notas_previstas[i]) for i in range(n))
mae = soma_erro_absoluto / n

# R² = 1 - (SSE / SST)
soma_total = sum((notas[i] - media_y)**2 for i in range(n))
r2 = 1 - (soma_erro_quadrado / soma_total)

print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")

# Gráfico 1: Reta de regressão
plt.figure(figsize=(10, 6))
plt.scatter(horas_estudo, notas, color='blue', label='Dados Originais')
plt.plot(horas_estudo, notas_previstas, color='red', label=f'Regressão (y={m:.2f}x + {b:.2f})')
plt.xlabel('Horas de Estudo')
plt.ylabel('Nota')
plt.title('Regressão Linear Simples: Nota vs Horas de Estudo')
plt.legend()
plt.grid(True) 
plt.show()

# Gráfico 2: Resíduos
residuos = [notas[i] - notas_previstas[i] for i in range(n)]

plt.figure(figsize=(10, 6))
plt.scatter(horas_estudo, residuos, color='purple')
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Horas de Estudo')
plt.ylabel('Resíduo')
plt.title('Resíduos da Regressão Linear')
plt.grid(True)
plt.show()

# Gráfico 3: Distribuição dos Resíduos
plt.figure(figsize=(10, 6))
sns.histplot(residuos, kde=True, color='green', bins=5)
plt.xlabel('Resíduo')
plt.title('Distribuição dos Resíduos')
plt.grid(True)
plt.show()