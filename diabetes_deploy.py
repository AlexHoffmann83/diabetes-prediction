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
st.title("Preditor de Diabetes com DecisionTreeClassifier")

# Criação de campos para entrada de dados
pregnancies = st.selectbox('Número de Gravidez', list(range(0, 21)))
glucose = st.number_input('Glicose', min_value = 40, max_value = 300, value = 80)
bloodPressure = st.number_input('Pressão Arterial Diastólica', min_value = 20, max_value = 120, value = 80)
skinThickness = st.number_input('PCT (espessura da prega cutânea do triceps)', min_value = 0, max_value = 500, value = 20)
insulin = st.number_input('Insulina', min_value = 0, max_value = 900, value = 79)
bmi = st.number_input('IMC (índice de massa corporal)', min_value = 0.0, max_value = 100.0, value = 31.9, step=0.1, format="%.1f")
age = st.selectbox('Idade', list(range(0, 101)))

# Botão para realizar previsões
if st.button('Prever Diabetes'):

    # Executa a função de pré-processamento de dados
    input_data = preprocess_input(pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, age)

    # Faz a previsão com o modelo
    prediction = predict(input_data)

    st.subheader('Resultado da Previsão:')
    st.success('✅ Diabetes Prevista: **Sim**' if prediction[0] == 1 else '❌ Diabetes Prevista: **Não**')

    st.markdown("---")

    st.markdown(
    """
    <div style='text-align: justify'>
    Este projeto é um objeto de estudo de Inteligência Artificial (Machine Learning) realizado com dados do Pima Indians Diabetes Database.<br>
    "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database"<br>
    Não são considerados aqui os dados genéticos e comportamentais que levaram este grupo objeto do estudo a desenvolverem grande índice de incidência de diabetes.<br>
    O resultado deste preditor, apesar de ter uma alta acurácia ao prever diabetes no grupo de estudo, pode não ser assertivo ao prever diabetes em outros 
    grupos étnicos, ou em grupos com hábitos diferentes das índias de ascendência Pima estudadas pelo nosso modelo de ML.<br><br>
    <strong>Entretanto, em caso de previsão positiva para diabetes, recomendamos fortemente que o usuário consulte um médico especialista.</strong>
    </div>
    """,
    unsafe_allow_html=True
    )