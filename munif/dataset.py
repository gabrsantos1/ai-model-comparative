import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# ==================================
# 1. CARREGAR DATASET
# ==================================

df = pd.read_csv("brasileirao.csv")

print("Shape:")
print(df.shape)

print("\nPrimeiras linhas:")
print(df.head())

# ==================================
# 2. CRIAR VARIÁVEL ALVO
# ==================================

def classificar_desempenho(posicao):
    if posicao <= 4:
        return "Elite"
    elif posicao <= 8:
        return "Competitivo"
    elif posicao <= 16:
        return "Intermediario"
    else:
        return "Rebaixado"

df["categoria"] = df["place"].apply(classificar_desempenho)

print("\nDistribuição das categorias:")
print(df["categoria"].value_counts())

# ==================================
# 3. GRÁFICO DAS CATEGORIAS
# ==================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="categoria",
    order=df["categoria"].value_counts().index
)

plt.title("Distribuição das Categorias")
plt.xlabel("Categoria")
plt.ylabel("Quantidade")

plt.show()

# ==================================
# 4. FEATURES E TARGET
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

y = df["categoria"]

# ==================================
# 5. DIVISÃO TREINO / TESTE
# ==================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==================================
# 6. NORMALIZAÇÃO PARA MLP
# ==================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================================
# 7. MODELO 1 - ÁRVORE DE DECISÃO
# ==================================

tree_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_test)

# ==================================
# 8. RESULTADOS - ÁRVORE
# ==================================

acc_tree = accuracy_score(
    y_test,
    y_pred_tree
)

f1_tree = f1_score(
    y_test,
    y_pred_tree,
    average="weighted"
)

print("\n==============================")
print("ÁRVORE DE DECISÃO")
print("==============================")

print(f"\nAccuracy: {acc_tree:.4f}")
print(f"F1-Score: {f1_tree:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_tree))

cm_tree = confusion_matrix(
    y_test,
    y_pred_tree
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm_tree,
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
    "Importancia": tree_model.feature_importances_
})

importancias = importancias.sort_values(
    by="Importancia",
    ascending=False
)

print("\nImportância das Variáveis:")
print(importancias)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=importancias,
    x="Importancia",
    y="Variavel"
)

plt.title("Importância das Variáveis")
plt.show()

# ==================================
# 10. MODELO 2 - SVM
# ==================================

svm_model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    random_state=42
)

svm_model.fit(
    X_train_scaled,
    y_train
)

y_pred_svm = svm_model.predict(
    X_test_scaled
)

# ==================================
# 11. RESULTADOS - SVM
# ==================================

acc_svm = accuracy_score(
    y_test,
    y_pred_svm
)

f1_svm = f1_score(
    y_test,
    y_pred_svm,
    average="weighted"
)

print("\n==============================")
print("SVM")
print("==============================")

print(f"\nAccuracy: {acc_svm:.4f}")
print(f"F1-Score: {f1_svm:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm))

cm_svm = confusion_matrix(
    y_test,
    y_pred_svm
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm_svm,
    annot=True,
    fmt="d",
    cmap="Greens"
)

plt.title("Matriz de Confusão - SVM")
plt.xlabel("Previsto")
plt.ylabel("Real")

plt.show()

# ==================================
# 11. COMPARAÇÃO DOS MODELOS
# ==================================

comparacao = pd.DataFrame({
    "Modelo": [
        "Árvore de Decisão",
        "SVM"
    ],
    "Accuracy": [
        acc_tree,
        acc_svm
    ],
    "F1-Score": [
        f1_tree,
        f1_svm
    ]
})

print("\nComparação dos Modelos:")
print(comparacao)

# Accuracy

plt.figure(figsize=(8, 5))

sns.barplot(
    data=comparacao,
    x="Modelo",
    y="Accuracy"
)

plt.title("Comparação de Accuracy")
plt.ylim(0, 1)

plt.show()

# F1 Score

plt.figure(figsize=(8, 5))

sns.barplot(
    data=comparacao,
    x="Modelo",
    y="F1-Score"
)

plt.title("Comparação de F1-Score")
plt.ylim(0, 1)

plt.show()

# ==================================
# 12. HEATMAP DE CORRELAÇÃO
# ==================================

corr = df[
    [
        "place",
        "points",
        "won",
        "draw",
        "loss",
        "goals",
        "goals_taken",
        "goals_diff"
    ]
].corr()

plt.figure(figsize=(10, 8))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlação Entre Variáveis")

plt.show()

# ==================================
# 13. SALVAR MODELOS
# ==================================

joblib.dump(
    tree_model,
    "modelo_arvore.pkl"
)

joblib.dump(
    svm_model,
    "modelo_svn.pkl"
)

joblib.dump(
    scaler,
    "scaler.pkl"
)

print("\nModelos salvos com sucesso!")