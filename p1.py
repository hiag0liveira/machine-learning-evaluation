import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import io
import sys
from contextlib import redirect_stdout
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams

# Ajusta fonte menor para caber mais texto nas figuras
rcParams.update({'font.size': 10})

# Importa funções do scikit-learn para divisão de dados, pré-processamento, modelagem e avaliação
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

# =============================================================================
# Função para transformar texto em uma página no PDF
# =============================================================================
def text_to_pdf(text, pdf, title='Texto'):
    """
    Cria uma figura de matplotlib com o conteúdo de 'text' e adiciona essa página no PDF.
    """
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    ax.text(0.0, 1.0, text, fontsize=10, ha='left', va='top', fontfamily='monospace')
    ax.set_title(title, pad=10, fontsize=12)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

# =============================================================================
# Leitura dos datasets e verificação inicial
# =============================================================================
# Defina os caminhos dos arquivos (ajuste se necessário)
car_file = "data/car+evaluation/car.data"
math_file = "data/student/student-mat.csv"
energy_file = "data/energy+efficiency/ENB2012_data.xlsx"

# Lista dos arquivos para verificação
files = [car_file, math_file, energy_file]
buffer_overall = io.StringIO()

with redirect_stdout(buffer_overall):
    print("# Verificação dos arquivos:")
    for file in files:
        if os.path.exists(file):
            print(f" \/ - Arquivo encontrado: {file}")
        else:
            print(f" X - Arquivo NÃO encontrado: {file}")
    print("\n" + "*"*30 + "\n")

# =============================================================================
# Leitura dos datasets
# =============================================================================
# Car Evaluation (ainda sem pré-processamento, com nomes de colunas definidos)
car_data = pd.read_csv(car_file,
                       names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"])
# Student Math Data
math_data = pd.read_csv(math_file, sep=";")
# Energy Efficiency Data
energy_data = pd.read_excel(energy_file)

# =============================================================================
# Função para exibir informações iniciais do dataset
# =============================================================================
def show_initial_info(df, title):
    """
    Exibe as primeiras linhas, dimensões e valores nulos do DataFrame.
    Retorna um texto com essas informações.
    """
    info_text = f"===== {title} =====\n\n"
    info_text += "Primeiras linhas:\n"
    info_text += df.head().to_string() + "\n\n"
    info_text += f"Dimensões (linhas, colunas): {df.shape}\n\n"
    info_text += "Valores nulos por coluna:\n"
    info_text += df.isnull().sum().to_string() + "\n"
    info_text += "\n" + "*"*30 + "\n\n"
    return info_text

buffer_overall.write(show_initial_info(car_data, "Car Evaluation Data"))
buffer_overall.write(show_initial_info(math_data, "Student Math Data"))
buffer_overall.write(show_initial_info(energy_data, "Energy Efficiency Data"))

# =============================================================================
# Pré-processamento: remoção de linhas com valores nulos
# =============================================================================
def preprocess_data(df, title, columns_to_check=None):
    """
    Realiza o pré-processamento removendo as linhas que contenham valores nulos.
    Se columns_to_check for informado, remove apenas com base nessas colunas.
    Retorna o DataFrame limpo, o texto informativo e a quantidade de linhas removidas.
    """
    text = f"===== Pré-processamento - {title} =====\n\n"
    rows_before = df.shape[0]
    text += f"Linhas antes: {rows_before}\n"
    if columns_to_check is not None:
        df_clean = df.dropna(subset=columns_to_check)
    else:
        df_clean = df.dropna()
    rows_after = df_clean.shape[0]
    rows_removed = rows_before - rows_after
    text += f"Linhas removidas (com valores nulos): {rows_removed}\n"
    text += f"Linhas após pré-processamento: {rows_after}\n"
    text += "\n" + "*"*30 + "\n\n"
    return df_clean, text

# Definir quais colunas verificar (se necessário, pode ser todas)
# No caso de car_data e math_data, supomos que não existam valores nulos; para energy_data, vamos verificar em todas.
car_data_clean, txt_car = preprocess_data(car_data, "Car Evaluation Data")
math_data_clean, txt_math = preprocess_data(math_data, "Student Math Data")
energy_data_clean, txt_energy = preprocess_data(energy_data, "Energy Efficiency Data")

buffer_overall.write(txt_car)
buffer_overall.write(txt_math)
buffer_overall.write(txt_energy)

# =============================================================================
# Pré-processamento específico para cada dataset
# =============================================================================
# Para o Car Evaluation, como todas as colunas são categóricas, usamos LabelEncoder
encoder = LabelEncoder()
for col in car_data_clean.columns:
    car_data_clean[col] = encoder.fit_transform(car_data_clean[col])
X_car = car_data_clean.drop("class", axis=1)
y_car = car_data_clean["class"]

# Para Student Math Data: removemos as colunas de notas e usamos o valor final para classificação
X_math = math_data_clean.drop(["G1", "G2", "G3"], axis=1)
y_math = pd.cut(math_data_clean["G3"], bins=3, labels=["low", "medium", "high"])
# Codifica colunas categóricas
def encode_categorical_columns(df):
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df
X_math = encode_categorical_columns(X_math)

# Para Energy Efficiency Data: cria uma classe com base em Y1
energy_data_clean["class"] = pd.cut(energy_data_clean["Y1"], bins=3, labels=["low", "medium", "high"])
X_energy = energy_data_clean.drop(["Y1", "Y2", "class"], axis=1)
y_energy = energy_data_clean["class"]
scaler = StandardScaler()
X_energy = scaler.fit_transform(X_energy)

# =============================================================================
# Divisão em treino e teste para cada dataset
# =============================================================================
def split_data(X, y, title):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    text = f"Divisão Treino/Teste - {title}:\n"
    text += f"X_train: {X_train.shape}, y_train: {np.array(y_train).shape}\n"
    text += f"X_test : {X_test.shape}, y_test : {np.array(y_test).shape}\n"
    text += "\n" + "*"*30 + "\n\n"
    return X_train, X_test, y_train, y_test, text

X_car_train, X_car_test, y_car_train, y_car_test, txt_split_car = split_data(X_car, y_car, "Car Evaluation")
X_math_train, X_math_test, y_math_train, y_math_test, txt_split_math = split_data(X_math, y_math, "Student Math")
X_energy_train, X_energy_test, y_energy_train, y_energy_test, txt_split_energy = split_data(X_energy, y_energy, "Energy Efficiency")

buffer_overall.write(txt_split_car)
buffer_overall.write(txt_split_math)
buffer_overall.write(txt_split_energy)

# =============================================================================
# Análise Descritiva: gráficos e estatísticas
# =============================================================================
def descriptive_analysis(df, title, pdf=None):
    """
    Realiza análise descritiva do DataFrame:
      - Exibe estatísticas descritivas e valores nulos (texto)
      - Gera histogramas, boxplots, heatmap e scatter plot (se aplicável)
    """
    report_text = f"===== {title} =====\n\n"
    report_text += "Estatísticas Descritivas:\n" + df.describe(include="all").to_string() + "\n\n"
    report_text += "\n" + "*"*30 + "\n\n"
    
    if pdf:
        text_to_pdf(report_text, pdf, title=f"Análise Descritiva - {title}")
    
    # Histograma (para colunas numéricas)
    numeric_cols = df.select_dtypes(include=[np.number])
    if not numeric_cols.empty:
        plt.figure(figsize=(8,6))
        numeric_cols.hist(bins=20, figsize=(12,8))
        plt.suptitle(f"Histogramas - {title}")
        if pdf:
            pdf.savefig()
        plt.close()
    
    # Boxplot
    if not numeric_cols.empty:
        plt.figure(figsize=(8,6))
        sns.boxplot(data=numeric_cols)
        plt.title(f"Boxplot - {title}")
        if pdf:
            pdf.savefig()
        plt.close()
    
    # Heatmap de correlação (se houver mais de uma coluna numérica)
    if numeric_cols.shape[1] > 1:
        plt.figure(figsize=(8,6))
        corr = numeric_cols.corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(f"Heatmap de Correlação - {title}")
        if pdf:
            pdf.savefig()
        plt.close()
    
    # Scatter plot (usando as duas primeiras colunas numéricas, se existirem)
    if numeric_cols.shape[1] >= 2:
        cols = numeric_cols.columns[:2]
        plt.figure(figsize=(8,6))
        sns.scatterplot(x=cols[0], y=cols[1], data=df)
        plt.title(f"Scatter Plot - {title} ({cols[0]} vs {cols[1]})")
        if pdf:
            pdf.savefig()
        plt.close()

# =============================================================================
# Modelagem: Treinamento, avaliação e apresentação dos resultados
# =============================================================================
def train_and_evaluate(X_train, X_test, y_train, y_test, dataset_name, pdf=None):
    """
    Treina três algoritmos de classificação e avalia usando Cross-validation e Split-Sample.
    Utiliza "precisão" como métrica de avaliação.
    """
    report_text = f"==== Avaliação - {dataset_name} ====\n\n"
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(kernel='linear', random_state=42)
    }
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Calcula precisão
        prec = precision_score(y_test, y_pred, average='weighted')
        cv_scores = cross_val_score(model, np.concatenate((X_train, X_test), axis=0),
                                    np.concatenate((y_train, y_test), axis=0), cv=5)
        cv_mean = np.mean(cv_scores)
        report_text += f"{name}:\n  Precisão: {prec:.4f}\n  CV (média): {cv_mean:.4f}\n"
        report_text += "*"*30 + "\n\n"
        results.append((name, prec, cv_mean))
        # Adiciona relatório de classificação (opcional)
        report_text += classification_report(y_test, y_pred) + "\n" + "*"*30 + "\n\n"
    
    if pdf:
        text_to_pdf(report_text, pdf, title=f"Resultados - {dataset_name}")
    
    # Gráfico comparativo
    if pdf:
        plt.figure(figsize=(8,6))
        bar_width = 0.35
        names = [r[0] for r in results]
        prec_values = [r[1] for r in results]
        cv_values = [r[2] for r in results]
        x_pos = np.arange(len(names))
        plt.bar(x_pos - bar_width/2, prec_values, width=bar_width, label="Precisão (Split)")
        plt.bar(x_pos + bar_width/2, cv_values, width=bar_width, label="CV (Média)")
        plt.xticks(x_pos, names)
        plt.ylabel("Métrica")
        plt.title(f"Comparativo de Resultados - {dataset_name}")
        plt.legend()
        pdf.savefig()
        plt.close()

# =============================================================================
# Geração do relatório final em PDF
# =============================================================================
with PdfPages("relatorio_classificacao.pdf") as pdf:
    # Primeiro, salvar o texto acumulado do buffer geral (leitura e splits)
    text_to_pdf(buffer_overall.getvalue(), pdf, title="Informações Iniciais dos Datasets")
    
    # Análise descritiva para cada dataset
    descriptive_analysis(car_data_clean, "Car Evaluation Data", pdf)
    descriptive_analysis(math_data_clean, "Student Math Data", pdf)
    descriptive_analysis(energy_data_clean, "Energy Efficiency Data", pdf)
    
    # Modelagem e avaliação para cada conjunto de dados
    train_and_evaluate(X_car_train, X_car_test, y_car_train, y_car_test, "Car Evaluation", pdf)
    train_and_evaluate(X_math_train, X_math_test, y_math_train, y_math_test, "Student Math", pdf)
    train_and_evaluate(X_energy_train, X_energy_test, y_energy_train, y_energy_test, "Energy Efficiency", pdf)

print("\nPDF gerado com sucesso: relatorio_classificacao.pdf")
