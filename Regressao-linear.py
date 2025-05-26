import math # Usaremos math.sqrt para o desvio padrão, se necessário, mas não para m e b diretamente.
import matplotlib.pyplot as plt # Para visualização

# Nossos dados de exemplo
horas_estudo = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
notas = [50, 55, 65, 70, 72, 80, 85, 88, 92, 95]

# Vamos garantir que temos a mesma quantidade de dados para X e Y
if len(horas_estudo) != len(notas):
    raise ValueError("As listas de horas de estudo e notas devem ter o mesmo tamanho.")

n = len(horas_estudo)

# 1. Calcular as médias de X e Y
media_x = sum(horas_estudo) / n
media_y = sum(notas) / n

print(f"Média de Horas de Estudo (X̄): {media_x:.2f}")
print(f"Média das Notas (Ȳ): {media_y:.2f}")

# 2. Calcular m (coeficiente angular)
# Usaremos a fórmula: m = Σ((Xi - X̄)(Yi - Ȳ)) / Σ((Xi - X̄)²)

numerador_m = 0
denominador_m = 0

for i in range(n):
    numerador_m += (horas_estudo[i] - media_x) * (notas[i] - media_y)
    denominador_m += (horas_estudo[i] - media_x)**2

if denominador_m == 0:
    raise ValueError("O denominador para o cálculo de 'm' é zero. Isso pode acontecer se todos os valores de X forem iguais.")

m = numerador_m / denominador_m
print(f"Coeficiente Angular (m): {m:.2f}")

# 3. Calcular b (intercepto linear)
# Usaremos a fórmula: b = Ȳ - mX̄
b = media_y - m * media_x
print(f"Intercepto Linear (b): {b:.2f}")

print(f"\nA equação da nossa reta de regressão é: Nota = {m:.2f} * Horas_Estudo + {b:.2f}")

# 4. Fazendo Previsões
# Vamos prever a nota para, por exemplo, 7.5 horas de estudo
horas_para_prever = 7.5
nota_prevista = m * horas_para_prever + b
print(f"Para {horas_para_prever} horas de estudo, a nota prevista é: {nota_prevista:.2f}")

# 5. Visualização dos dados e da reta de regressão
plt.figure(figsize=(10, 6))
# Pontos de dados originais
plt.scatter(horas_estudo, notas, color='blue', label='Dados Originais (Nota vs Horas)')

# Linha de regressão
# Para desenhar a linha, precisamos de alguns pontos X e seus Ys previstos
x_reta = [min(horas_estudo), max(horas_estudo)] # Pegamos o mínimo e máximo das horas para definir a linha
y_reta = [m * x_val + b for x_val in x_reta]

plt.plot(x_reta, y_reta, color='red', label=f'Linha de Regressão (y={m:.2f}x + {b:.2f})')

plt.xlabel("Horas de Estudo")
plt.ylabel("Nota")
plt.title("Regressão Linear Simples: Nota vs Horas de Estudo")
plt.legend()
plt.grid(True)
plt.show()