import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Carregar dados do arquivo XLSX
try:
    df = pd.read_excel('study_data.xlsx') 
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

n = len(df)

# Inicialização dos coeficientes
m = 0
b = 0

# Hiperparâmetros
learning_rate = 0.01
epochs = 1000

# Gradiente descendente
for epoch in range(epochs):
    erro_total = 0
    dm = 0
    db = 0

    for x, y in zip(horas_estudo, notas):
        y_pred = m * x + b
        erro = y - y_pred
        erro_total += erro ** 2
        dm += -2 * x * erro
        db += -2 * erro

    m -= learning_rate * (dm / n)
    b -= learning_rate * (db / n)

    if epoch % 100 == 0:
        print(f'Época {epoch}: Erro médio quadrático = {erro_total / n:.4f}')

# Resultado da regressão
print(f"\nCoeficiente Angular (m): {m:.4f}")
print(f"Intercepto Linear (b): {b:.4f}")
print(f"Equação: Nota = {m:.4f} * Horas_Estudo + {b:.4f}\n")

# Previsão
horas_para_prever = float(input("\nDigite um valor horas para prever a nota!\n-> "))
nota_prevista = m * horas_para_prever + b
print(f"Para {horas_para_prever} horas de estudo, nota prevista: {nota_prevista:.2f}")

notas_previstas = [m * x + b for x in horas_estudo]

# Métricas
soma_erro_quadrado = sum((notas[i] - notas_previstas[i])**2 for i in range(n))
mse = soma_erro_quadrado / n
rmse = math.sqrt(mse)
soma_erro_absoluto = sum(abs(notas[i] - notas_previstas[i]) for i in range(n))
mae = soma_erro_absoluto / n
media_y = sum(notas) / n
soma_total = sum((notas[i] - media_y)**2 for i in range(n))
r2 = 1 - (soma_erro_quadrado / soma_total)

print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")

# Gráficos
plt.figure(figsize=(10, 6))
plt.scatter(horas_estudo, notas, color='blue', label='Dados Originais')
plt.plot(horas_estudo, notas_previstas, color='red', label=f'Regressão (y={m:.2f}x + {b:.2f})')
plt.xlabel('Horas de Estudo')
plt.ylabel('Nota')
plt.title('Regressão Linear com Gradiente Descendente')
plt.legend()
plt.grid(True) 
plt.show()

# Gráfico de resíduos
residuos = [notas[i] - notas_previstas[i] for i in range(n)]

plt.figure(figsize=(10, 6))
plt.scatter(horas_estudo, residuos, color='purple')
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Horas de Estudo')
plt.ylabel('Resíduo')
plt.title('Resíduos da Regressão Linear')
plt.grid(True)
plt.show()

# Distribuição dos resíduos
plt.figure(figsize=(10, 6))
sns.histplot(residuos, kde=True, color='green')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero (Ideal)')
plt.xlabel('Resíduo')
plt.title('Distribuição dos Resíduos')
plt.legend()
plt.grid(True)
plt.show()
