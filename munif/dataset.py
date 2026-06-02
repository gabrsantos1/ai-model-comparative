import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import seaborn as sns
import matplotlib.pyplot as plt

# ==================================
# 1. CARREGAR DATASET
# ==================================

df = pd.read_csv("brasileirao.csv")

print("Shape:")
print(df.shape)

print("\nColunas:")
print(df.columns.tolist())

print("\nPrimeiras linhas:")
print(df.head())

# ==================================
# 2. CRIAR VARIÁVEL ALVO
# ==================================

# Rebaixados = posições 17,18,19,20

df["rebaixado"] = (df["place"] >= 17).astype(int)

print("\nQuantidade de rebaixados:")
print(df["rebaixado"].value_counts())

# ==================================
# 3. SELECIONAR FEATURES
# ==================================

X = df[
    [
        "points",
        "won",
        "draw",
        "loss",
        "goals",
        "goals_taken",
        "goals_diff"
    ]
]

y = df["rebaixado"]

# ==================================
# 4. DIVISÃO TREINO / TESTE
# ==================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==================================
# 5. TREINAR MODELO
# ==================================

modelo = DecisionTreeClassifier(
    random_state=42
)

modelo.fit(X_train, y_train)

# ==================================
# 6. PREVISÕES
# ==================================

y_pred = modelo.predict(X_test)

# ==================================
# 7. MÉTRICAS
# ==================================

acc = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(acc)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# ==================================
# 8. MATRIZ DE CONFUSÃO
# ==================================

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Matriz de Confusão - Árvore de Decisão")
plt.xlabel("Previsto")
plt.ylabel("Real")

plt.show()

# ==================================
# 9. IMPORTÂNCIA DAS VARIÁVEIS
# ==================================

importancias = pd.DataFrame({
    "Variavel": X.columns,
    "Importancia": modelo.feature_importances_
})

importancias = importancias.sort_values(
    by="Importancia",
    ascending=False
)

print("\nImportância das variáveis:")
print(importancias)

plt.figure(figsize=(8,4))

sns.barplot(
    data=importancias,
    x="Importancia",
    y="Variavel"
)

#times q mais cairam
rebaixamentos = df[df["place"] >= 17]["team"].value_counts()
print("Qual time mais caiu?", rebaixamentos)

#times q mais ficaram no g4
g4 = df[df["place"] <= 4]["team"].value_counts()
print("Time g4",g4)

#rendimento de times especificos
palmeiras = df[df["team"] == "Palmeiras"].sort_values("season")
print(palmeiras[["season", "place"]])

plt.title("Importância das Variáveis")
plt.show()

# evolucao de posicao por time
import matplotlib.pyplot as plt

team_name = "Vasco"

team_df = df[df["team"] == team_name].sort_values("season")

plt.plot(team_df["season"], team_df["place"], marker="o")

plt.gca().invert_yaxis()  # posição 1 no topo
plt.title(f"Evolução do {team_name}")
plt.xlabel("Ano")
plt.ylabel("Posição")
plt.show()

#ranking geral dos q mais cairam
top_rebaixados = df[df["place"] >= 17]["team"].value_counts()

print(top_rebaixados)

#times mais consistentes no campeonato
consistencia = df.groupby("team")["place"].std().sort_values()

print(consistencia.head(10))

#media de posicao geral
media_posicao = df.groupby("team")["place"].mean().sort_values()

print(media_posicao)