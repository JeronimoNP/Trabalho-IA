import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# 1 - Carregar os dados
df = pd.read_excel('C:\\Users\\famil\\OneDrive\\Documentos\\Projetos\\Trabalho-IA\\RMA\\Pasta1.xlsx')
# Converter a variável categórica para numérica
df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

# 2 - Criar variável alvo binária (exemplo: Aprovado >= 600 pontos)
df['Aprovado'] = df['Performance Index'].apply(lambda x: 1 if x >= 600 else 0)

# 3 - Separar X e y
X = df.drop(['Performance Index', 'Aprovado'], axis=1)
y = df['Aprovado']

# 4 - Normalizar
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 5 - Divisão 70/10/20
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42, stratify=y_temp)

# 6 - Criar o modelo RMA (Rede Neural) com Keras
model = Sequential()
model.add(Dense(8, activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 7 - Compilar o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 8 - Treinar com validação
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_val, y_val),
    verbose=1
)

# 9 - Avaliar no conjunto de teste
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'\nAcurácia no Teste: {accuracy:.4f}')

# 10 - Previsões
y_pred_prob = model.predict(X_test).ravel()  # Probabilidades
y_pred = (y_pred_prob > 0.5).astype(int)      # Converte para 0 ou 1

# ---- Previsão com valores fixos no código ----

# Valores de entrada numero 9142:
# [Horas Estudadas, Pontuação Anterior, Atividade Extracurricular, Horas de Sono, Número de Questões Praticadas]

nova_entrada = pd.DataFrame([[9, 96, 0, 8, 3]], columns=[
    'Hours Studied',
    'Previous Scores',
    'Extracurricular Activities',
    'Sleep Hours',
    'Sample Question Papers Practiced'
])

# Normalizar igual foi feito no treinamento
nova_entrada = scaler.transform(nova_entrada)

# Fazer a previsão
probabilidade = model.predict(nova_entrada)[0][0]
classe_prevista = 1 if probabilidade >= 0.5 else 0

print(f"Probabilidade de passar: {probabilidade * 100:.2f}%")
print(f"Previsão: {'Passou' if classe_prevista == 1 else 'Não passou'}")


# 11 - Matriz de Confusão
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# 12 - Relatório de Classificação: Precisão, Recall, F1, etc
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# 13 - Curva ROC e AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
print(f"\nAUC-ROC: {roc_auc:.4f}")

# 14 - Plot da Curva ROC
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

# 15 - Plot da Perda (Loss) durante o treinamento
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Perda durante o Treinamento')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.show()
