# Preditor de Diabetes com Machine Learning

Este projeto é um estudo prático de Inteligência Artificial aplicado à área da saúde. Utilizamos o **Pima Indians Diabetes Database** para treinar um modelo de aprendizado de máquina capaz de prever a ocorrência de diabetes com base em dados clínicos.

---

## Objetivo

Criar um **modelo de classificação** para prever a probabilidade de uma pessoa apresentar diabetes, utilizando variáveis clínicas de fácil obtenção, e disponibilizar o resultado final como uma aplicação web interativa com **Streamlit**.

---

## Conjunto de Dados

- **Fonte:** Pima Indians Diabetes Database (UCI Machine Learning Repository)
- **Amostras:** 768 mulheres de origem Pima (grupo indígena dos EUA)
- **Variáveis utilizadas:**
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - Age
  - **Exclusão:** A variável *DiabetesPedigreeFunction* foi retirada do modelo final, pois não é uma medida comum na prática clínica cotidiana.

---

## Técnicas Aplicadas

- Pré-processamento de dados (remoção de entradas inválidas e normalização com `StandardScaler`)
- Comparação de modelos: `DecisionTreeClassifier` vs `RandomForestClassifier`
- Validação cruzada (5-fold)
- Análise de overfitting (comparação treino vs teste)
- Ajuste de hiperparâmetros
- Deploy com **Streamlit**

---

## Modelo Final Escolhido

### DecisionTreeClassifier
- `max_depth=4`
- `class_weight='balanced'`
- `random_state=42`

** Motivo da escolha:**  
Apresentou o melhor equilíbrio entre desempenho no conjunto de treino e teste, sem overfitting, e maior recall para casos positivos (diabetes), que é o principal objetivo do projeto.

---

## Desempenho do Modelo

### Conjunto de Teste:

| Classe | Precisão | Recall | F1-Score |
|--------|----------|--------|----------|
| 0 (não diabético) | 83% | 77% | 0.80 |
| 1 (diabético)     | 61% | 70% | 0.66 |

*Acurácia geral:* **74,7%**

---

## Como executar o projeto

### 1. Clone o repositório:
git clone https://github.com/seu-usuario/diabetes-prediction-streamlit-ml.git
cd diabetes-prediction-streamlit-ml

2. Crie um ambiente virtual e ative:
conda create -n diabetes-ml python=3.11
conda activate diabetes-ml

3. Instale as dependências:
pip install -r requirements.txt

4. Rode a aplicação:
streamlit run app/diabetes_deploy.py
📸 Imagem da aplicação
(Adicione aqui um print do Streamlit rodando, se quiser)
![app screenshot](app_screenshot.png)

Observações Importantes
Este projeto é apenas um exemplo didático de aplicação de Machine Learning.
O modelo foi treinado com base em um grupo étnico específico e não deve ser utilizado para diagnóstico médico real.

Autor:
Alex Hoffmann
GitHub