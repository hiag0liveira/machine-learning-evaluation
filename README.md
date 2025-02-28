# **Avaliação de Modelos de Aprendizado de Máquina**

Este repositório contém um projeto de **Machine Learning**, onde realizamos a **análise, pré-processamento e avaliação de modelos de classificação** utilizando três diferentes bases de dados e três algoritmos distintos. O objetivo principal é comparar o desempenho dos modelos utilizando **validação cruzada** e **divisão treino/teste**.

---

## **Descrição do Projeto**

O foco deste trabalho é avaliar a capacidade preditiva de diferentes algoritmos de aprendizado de máquina em **três conjuntos de dados distintos**. O fluxo de trabalho inclui:

1. **Coleta e exploração** de dados de diferentes fontes.
2. **Pré-processamento**, incluindo tratamento de valores nulos e codificação de variáveis categóricas.
3. **Aplicação de três algoritmos de classificação**:
   - Random Forest
   - Decision Tree
   - SVM (Support Vector Machine)
4. **Avaliação dos modelos** por meio de:
   - **Validação Cruzada (Cross-validation)**
   - **Divisão Treino/Teste (Split-Sample Validation)**
5. **Geração de relatórios** contendo:
   - Histogramas, boxplots e scatter plots.
   - Matrizes de confusão e relatórios de classificação.
   - Comparação entre os modelos com base na métrica de **precisão (Precision).**

---

## **Bases de Dados Utilizadas**

Os conjuntos de dados foram obtidos de fontes públicas, como **UCI Machine Learning Repository** e **OpenML**. As bases utilizadas foram:

1. **Car Evaluation**

   - Contém informações sobre características de carros e uma **classificação final** sobre quão aceitáveis eles são.
   - Atributos como **preço de compra, custo de manutenção, número de portas, segurança, entre outros**.
   - **Objetivo:** Classificar o carro em categorias como **"inaceitável", "aceitável", "bom" ou "muito bom"**.

2. **Student Math Performance**

   - Conjunto de dados sobre **alunos de matemática**, incluindo informações sobre **notas, ambiente familiar e atividades extracurriculares**.
   - O modelo foi treinado para prever o **desempenho final do aluno**, agrupando as notas em **três categorias**: **baixo, médio e alto desempenho**.

3. **Energy Efficiency**
   - Conjunto de dados contendo **informações sobre edifícios residenciais** e seu consumo energético.
   - O objetivo foi prever **a eficiência energética** com base em atributos como **área, altura, orientação e isolamento térmico**.
   - A variável alvo foi convertida em **três categorias**: **baixo, médio e alto consumo energético**.

---

## **Estrutura do Repositório**

A estrutura do projeto segue a seguinte organização:

```
machine-learning-evaluation/
├── data/
│   ├── car+evaluation.csv
│   ├── student-math.csv
│   ├── energy-efficiency.xlsx
├── p1.py                           # Script principal (leitura, análise e avaliação dos modelos)
├── relatorio_classificacao.pdf     # Relatório final contendo gráficos e métricas
├── README.md                       # Este arquivo
└── requirements.txt                # Lista de dependências do projeto
```

- **data/** → Contém os datasets utilizados no projeto.
- **p1.py** → Contém os scripts Python responsáveis por cada etapa do pipeline.
- **relatorio_classificacao/** → Contém o relatório final gerado automaticamente.
- **requirements.txt** → Lista de bibliotecas necessárias para rodar o projeto.

---

## **Como Executar o Projeto**

### **1. Clonar o Repositório**

No terminal, execute:

```bash
git clone https://github.com/hiag0liveira/machine-learning-evaluation.git
cd machine-learning-evaluation
```

### **2. Criar e Ativar o Ambiente Virtual (Recomendado)**

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### **3. Instalar as Dependências**

Com o ambiente virtual ativo, instale as bibliotecas necessárias:

```bash
pip install -r requirements.txt
```

Caso o `requirements.txt` não esteja disponível, instale manualmente:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn openpyxl
```

### **4. Executar o Script Principal**

Para rodar a análise e gerar os gráficos:

```bash
python p1.py
```

O relatório final será salvo automaticamente como `relatorio_classificacao.pdf`.

---

## **Modelos Utilizados**

Foram utilizados **três algoritmos de classificação** para comparar o desempenho preditivo:

1. **Random Forest**

   - Algoritmo baseado em múltiplas árvores de decisão.
   - Bom para capturar padrões complexos e lidar com dados categóricos.

2. **Decision Tree**

   - Modelo mais simples baseado em regras de decisão.
   - Fácil de interpretar, mas pode ser propenso a overfitting.

3. **SVM (Support Vector Machine)**
   - Algoritmo que busca encontrar um hiperplano ótimo para separar as classes.
   - Funciona bem para dados lineares e de alta dimensionalidade.

---

## **Métricas de Avaliação**

Os modelos foram avaliados utilizando duas abordagens:

1. **Divisão Treino/Teste (Split-Sample Validation)**

   - Separação dos dados em **70% para treino** e **30% para teste**.
   - Avaliação do desempenho do modelo em dados não vistos.

2. **Validação Cruzada (Cross-Validation - 5 folds)**
   - Os dados foram divididos em 5 partes (folds), onde cada modelo foi treinado e testado 5 vezes em diferentes divisões do conjunto de dados.
   - Isso garante uma **avaliação mais estável** e reduz a dependência da divisão inicial dos dados.

A métrica principal escolhida foi **Precisão (Precision)**, mas também foram analisados **acurácia, recall e matriz de confusão**.

---

## **Conclusões**

- A combinação de **Split-Sample Validation** e **Cross-Validation** garantiu uma avaliação mais confiável dos modelos.
- O **Random Forest** foi o modelo com melhor desempenho geral.
- Modelos baseados em árvores foram mais eficientes para bases com atributos categóricos.
- O SVM teve melhor desempenho em dados contínuos (Energy Efficiency) e pior desempenho em bases categóricas.
