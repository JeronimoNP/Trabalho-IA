# Calculo base e manipulação de dados
import pandas as pd
import numpy as np
# Visualização de dados
import seaborn as sns
import matplotlib.pyplot as plt
# RMA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Carregar os dados
df = pd.read_excel('C:\\Users\\famil\\OneDrive\\Documentos\\Projetos\\Trabalho-IA\\RMA\\Pasta1.xlsx')

# Tratamento de valores ausentes
for coluna in df.columns:
    if df[coluna].dtype in [np.float64, np.int64]:
        df[coluna].fillna(df[coluna].mean(), inplace=True)
    else:
        df[coluna].fillna(df[coluna].mode()[0], inplace=True)

# Converter a variável categórica para numérica
df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

# Criar variável alvo binária (Exemplo: aprovado se >= 600 pontos)
df['Aprovado'] = df['Performance Index'].apply(lambda x: 1 if x >= 600 else 0)

# Separar X e y
X = df.drop(['Performance Index', 'Aprovado'], axis=1)
y = df['Aprovado']

# Normalizar os dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir os dados: 70% treino, 10% validação, 20% teste
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42, stratify=y_temp)

# Criar o modelo (Rede Neural)
model = Sequential()
model.add(Dense(8, activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compilar o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinar com validação
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_val, y_val),
    verbose=1
)

# Avaliar o modelo no conjunto de teste
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'\nAcurácia no Teste: {accuracy:.4f}')

# Fazer as previsões no conjunto de teste
y_pred_prob = model.predict(X_test).ravel()                    # Probabilidades
y_pred = (y_pred_prob > 0.5).astype(int)                      # Converte para 0 ou 1

# Previsão para um caso específico (exemplo de entrada)
nova_entrada = pd.DataFrame([[9, 96, 0, 8, 3]], columns=[
    'Hours Studied',
    'Previous Scores',
    'Extracurricular Activities',
    'Sleep Hours',
    'Sample Question Papers Practiced'
])

# Normalizar a nova entrada igual ao treino
nova_entrada = scaler.transform(nova_entrada)

# Fazer a previsão
probabilidade = model.predict(nova_entrada)[0][0]
classe_prevista = 1 if probabilidade >= 0.5 else 0
print(f"\nProbabilidade de passar: {probabilidade * 100:.2f}%")
print(f"Previsão: {'Passou' if classe_prevista == 1 else 'Não passou'}")

# Informações sobre o dataset
print("\nInformações sobre o dataset:")
print(f"Quantidade total de dados: {len(df)}")
print(f"Número de variáveis de entrada: {X.shape[1]}")

# Análise exploratória
print('\nPrimeiras linhas do dataset:')
print(df.head())
print('\nInformações do dataset:')
df.info()
print('\nEstatísticas descritivas:')
print(df.describe())

# Matriz de Confusão
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# Calcula a matriz de confusão
cm = confusion_matrix(y_test, y_pred)

# Cria a figura do gráfico
plt.figure(figsize=(8, 6))

# Cria o heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Não Passou (Prev)', 'Passou (Prev)'],
            yticklabels=['Não Passou (Real)', 'Passou (Real)'])

# Adiciona os títulos e legendas
plt.title('Matriz de Confusão', fontsize=16)
plt.ylabel('Classe Real', fontsize=12)
plt.xlabel('Classe Prevista', fontsize=12)

# Mostra o gráfico
plt.show()

# Relatório de Classificação (Precisão, Recall, F1-Score, etc)
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Curva ROC e AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
print(f"\nAUC-ROC: {roc_auc:.4f}")

# Plot da Curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

# Plot da Perda (Loss) durante o treinamento
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Perda durante o Treinamento')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Comparação Acertos vs Erros nas primeiras 100 amostras de teste
num_amostras = min(100, len(y_test))
y_real = np.array(y_test)[:num_amostras]
y_prev = y_pred[:num_amostras]

plt.figure(figsize=(12, 6))

for idx in range(num_amostras):
    real = y_real[idx]
    prev = y_prev[idx]
    
    if real == prev:
        # Acerto → Ponto verde
        plt.scatter(idx, real, color='green', label='Acerto' if idx == 0 else "")
    else:
        plt.scatter(idx, prev, color='red', label='Previsto (Erro)' if idx == 0 else "")

plt.title('Acertos e Erros nas Primeiras 100 Amostras de Teste')
plt.xlabel('Amostra')
plt.ylabel('Classe (0 = Não Passou, 1 = Passou)')
plt.legend()
plt.show()