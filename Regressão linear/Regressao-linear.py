import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Carregar dados do arquivo XLSX
try:
    df = pd.read_excel('study_data.xlsx') 
    horas_estudo = df['Hours Studied'].tolist() 
    notas = df['Performance Index'].tolist() 
except FileNotFoundError:
    print("Erro: O arquivo 'study_data.xlsx' não foi encontrado. Verifique o nome e o caminho do arquivo.")
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

# Divisão treino/teste (70% treino, 30% teste)
data = list(zip(horas_estudo, notas))
random.shuffle(data)

split_index = int(0.7 * len(data))
train_data = data[:split_index]
test_data = data[split_index:]

horas_treino = [x[0] for x in train_data]
notas_treino = [x[1] for x in train_data]
horas_teste = [x[0] for x in test_data]
notas_teste = [x[1] for x in test_data]

n = len(horas_treino)

# Inicialização dos coeficientes
m = 0
b = 0

# Hiperparâmetros
learning_rate = 0.01
epochs = 1000

# Gradiente descendente (treinamento)
for epoch in range(epochs):
    erro_total = 0
    dm = 0
    db = 0

    for x, y in zip(horas_treino, notas_treino):
        y_pred = m * x + b
        erro = y - y_pred
        erro_total += erro ** 2
        dm += -2 * x * erro
        db += -2 * erro

    m -= learning_rate * (dm / n)
    b -= learning_rate * (db / n)

    if epoch % 100 == 0:
        print(f'Época {epoch}: Erro médio quadrático (Treino) = {erro_total / n:.4f}')

# Resultado da regressão
print(f"\nCoeficiente Angular (m): {m:.4f}")
print(f"Intercepto Linear (b): {b:.4f}")
print(f"Equação: Nota = {m:.4f} * Horas_Estudo + {b:.4f}\n")

# Previsão com dados de teste
notas_previstas = [m * x + b for x in horas_teste]

# Métricas (com base nos dados de teste)
n_teste = len(notas_teste)
soma_erro_quadrado = sum((notas_teste[i] - notas_previstas[i])**2 for i in range(n_teste))
mse = soma_erro_quadrado / n_teste
rmse = math.sqrt(mse)
mae = sum(abs(notas_teste[i] - notas_previstas[i]) for i in range(n_teste)) / n_teste
media_y = sum(notas_teste) / n_teste
soma_total = sum((notas_teste[i] - media_y)**2 for i in range(n_teste))
r2 = 1 - (soma_erro_quadrado / soma_total)

print(f"MSE (Teste): {mse:.2f}")
print(f"RMSE (Teste): {rmse:.2f}")
print(f"MAE (Teste): {mae:.2f}")
print(f"R² (Teste): {r2:.4f}")

# Previsão manual
horas_para_prever = float(input("\nDigite um valor de horas para prever a nota:\n-> "))
nota_prevista = m * horas_para_prever + b
print(f"Para {horas_para_prever} horas de estudo, nota prevista: {nota_prevista:.2f}")

# Gráficos
plt.figure(figsize=(10, 6))
plt.scatter(horas_teste, notas_teste, color='blue', label='Dados Reais (Teste)')
plt.plot(horas_teste, notas_previstas, color='red', label=f'Regressão (y={m:.2f}x + {b:.2f})')
plt.xlabel('Horas de Estudo')
plt.ylabel('Nota')
plt.title('Regressão Linear com Gradiente Descendente (Teste)')
plt.legend()
plt.grid(True) 
plt.show()

# Gráfico de resíduos
residuos = [notas_teste[i] - notas_previstas[i] for i in range(n_teste)]
plt.figure(figsize=(10, 6))
plt.scatter(horas_teste, residuos, color='purple')
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Horas de Estudo')
plt.ylabel('Resíduo')
plt.title('Resíduos da Regressão Linear (Teste)')
plt.grid(True)
plt.show()

# Distribuição dos resíduos
plt.figure(figsize=(10, 6))
sns.histplot(residuos, kde=True, color='green')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero (Ideal)')
plt.xlabel('Resíduo')
plt.title('Distribuição dos Resíduos (Teste)')
plt.legend()
plt.grid(True)
plt.show()

