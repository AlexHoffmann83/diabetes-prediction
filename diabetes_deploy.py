# Deploy do Modelo

# Para rodar do deploy, escrever no terminal: streamlit run diabetes_deploy.py

# Instale o streamlit: pip install streamlit

# Imports
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Carregar o modelo e o scaler
modelo = joblib.load('modelos/modelo_decision_tree_classifier.pkl')
scaler = joblib.load('modelos/scaler_standard.pkl')

# Função para pré-processar os dados de entrada
# As colunas devem ser exatamente as mesmas usadas durante o treinamento
def preprocess_input(pregnancies, 
                     glucose, 
                     bloodPressure, 
                     skinThickness, 
                     Insulin,
                     bmi, 
                     age):
    
    # Dataframe
    data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [bloodPressure],
        'SkinThickness': [skinThickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'Age': [age],
    })

    # Lista de colunas
    numeric_cols = ['Pregnancies', 
                    'Glucose', 
                    'BloodPressure', 
                    'SkinThickness', 
                    'Insulin',
                    'BMI',
                    'Age']

    # Aplicando padronização
    data[numeric_cols] = scaler.transform(data[numeric_cols])

    return data

# Função para fazer previsões
def predict(data):
    prediction = modelo.predict(data)
    return prediction

# Interface do Streamlit
#st.title("Preditor de Diabetes com Inteligência Artificial")
st.markdown("""
<h1 style='text-align: center; color: #2c3e50;'>
    Preditor de Diabetes com <br> Inteligência Artificial
</h1>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div style='padding: 10px; border: 2px solid #d9534f; border-radius: 5px; background-color: #f2dede; color: #a94442'>
<b>Aviso Importante:</b> Este aplicativo não é um dispositivo médico.
É um modelo preditivo desenvolvido com base em dados públicos (Pima Indians Diabetes Dataset),
e deve ser usado apenas para fins educacionais ou exploratórios. <br><br>
Este resultado não substitui a avaliação de um profissional da saúde. Em caso de suspeita de diabetes, procure atendimento médico.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

with st.expander("📈 Sobre o modelo de predição"):
    st.markdown("""
    - Modelo utilizado: **Árvore de Decisão (Decision Tree Classifier)**.
    - Acurácia geral obtida na validação: **74,68%**
    - Base de dados: [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).
    - Métricas calculadas usando validação de dados com divisão 80/20.
    - O desenvolvimento científico de validação de dados, escolha do algorítmo de Machine Learning e comentários pertinentes, podem ser visualizados no Github: [Repositório do projeto](https://github.com/AlexHoffmann83/diabetes-prediction).
    """)


# Criação de campos para entrada de dados
pregnancies = st.selectbox('Número de Gravidez', list(range(0, 21)))
glucose = st.number_input('Glicose', min_value = 40, max_value = 300, value = 80)
bloodPressure = st.number_input('Pressão Arterial Diastólica', min_value = 40, max_value = 150, value = 80)
skinThickness = st.number_input('PCT (espessura da prega cutânea do triceps)', min_value = 0, max_value = 500, value = 20)
insulin = st.number_input('Insulina', min_value = 0, max_value = 900, value = 79)
bmi = st.number_input('IMC (índice de massa corporal)', min_value = 0.0, max_value = 100.0, value = 31.9, step=0.1, format="%.1f")
age = st.selectbox('Idade', list(range(0, 101)), index=20)


# Botão para realizar previsões
if st.button('Prever Diabetes'):

    # Executa a função de pré-processamento de dados
    input_data = preprocess_input(pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, age)

    # Faz a previsão com o modelo
    prediction = predict(input_data)

    st.markdown('### Resultado da Previsão:')
    st.success('✅ Diabetes Prevista: **Sim**' if prediction[0] == 1 else '❌ Diabetes Prevista: **Não**')
    if prediction== 1:
        st.markdown("### **Risco elevado de diabetes**")
        st.warning("Recomenda-se procurar orientação médica para investigação detalhada.")
    else:
        st.markdown("### **Baixo risco identificado**")
        st.info("Mesmo com baixo risco, hábitos saudáveis e exames regulares são fundamentais para prevenção.")

st.markdown("---")

st.markdown(
"""
<div style='text-align: justify'>
Este projeto é um objeto de estudo de Inteligência Artificial (Machine Learning) realizado com dados do Pima Indians Diabetes Database.<br>
Base de dados: <a href='https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database' target='_blank'>Pima Indians Diabetes Database</a><br>
<br>
A base de dados utilizada neste projeto, foi escolhida por ser uma das mais amplamente utilizadas em pesquisas de aprendizado de máquina para o estudo de diabetes. Ela contém informações clínicas de mulheres de ascendência Pima, um grupo indígena norte-americano que apresenta alta incidência da doença, o que a torna valiosa para modelos preditivos. Por sua estrutura bem organizada, alta incidência de casos positivos de diabetes e amplo uso acadêmico, essa base é referência em benchmark para algoritmos de classificação.
<br><br>
⚠️ Atenção: Apesar da boa acurácia, o modelo não considera fatores clínicos adicionais
como histórico familiar, hábitos alimentares, nível de estresse ou exames laboratoriais não presentes na base original. <br><br>
O resultado deste preditor, apesar de ter uma alta acurácia ao prever diabetes no grupo de estudo, pode não ser assertivo ao prever diabetes em outros 
grupos étnicos, ou em grupos com hábitos diferentes das índias de ascendência Pima estudadas pelo nosso modelo de ML.<br><br>
<strong>Entretanto, em caso de previsão positiva para diabetes, recomendamos fortemente que o usuário consulte um médico especialista.</strong>
</div>
""",
unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("**Autor:** Alex Hoffmann  \nGitHub: [@AlexHoffmann83](https://github.com/AlexHoffmann83)")
