import streamlit as st
import pandas as pd
import joblib

model = joblib.load("modelo.pkl")

st.set_page_config(page_title="Titanic Survival", page_icon="🚢")
st.title("🚢 Titanic — Previsão de Sobrevivência")
st.markdown("Preencha os dados do passageiro para prever se ele sobreviveria ao naufrágio.")

pclass = st.selectbox("Classe do Navio", [1, 2, 3], format_func=lambda x: f"{x}ª Classe")
sex = st.radio("Sexo", ["Masculino", "Feminino"], horizontal=True)
age = st.slider("Idade", min_value=1, max_value=80, value=30)
sibsp = st.number_input("Número de Irmãos / Cônjuges a bordo", min_value=0, max_value=10, value=0)
fare = st.number_input("Tarifa Paga (£)", min_value=0.0, max_value=600.0, value=32.0, step=1.0)

if st.button("Prever Sobrevivência", use_container_width=True):
    sex_enc = 1 if sex == "Feminino" else 0

    input_data = pd.DataFrame([[pclass, sex_enc, age, sibsp, fare]],
                               columns=["Pclass", "Sex", "Age", "SibSp", "Fare"])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction]

    if prediction == 1:
        st.success(f"✅ O passageiro **SOBREVIVERIA** com {probability:.0%} de probabilidade.")
    else:
        st.error(f"❌ O passageiro **NÃO sobreviveria** com {probability:.0%} de probabilidade.")
