import streamlit as st
import pandas as pd
import joblib

model = joblib.load("modelo.pkl")
le_region = joblib.load("le_region.pkl")
le_model = joblib.load("le_model.pkl")

st.set_page_config(page_title="BMW Sales Predictor", page_icon="🚗")
st.title("BMW — Previsão de Unidades Vendidas")
st.markdown("Preencha os campos abaixo para prever quantas unidades serão vendidas.")

col1, col2 = st.columns(2)

with col1:
    year = st.selectbox("Ano", list(range(2018, 2026)))
    month = st.selectbox("Mês", list(range(1, 13)))
    region = st.selectbox("Região", le_region.classes_.tolist())
    model_name = st.selectbox("Modelo BMW", le_model.classes_.tolist())

with col2:
    avg_price = st.number_input("Preço Médio (EUR)", min_value=20000, max_value=200000, value=55000, step=1000)
    bev_share = st.slider("Participação BEV (0 a 1)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    gdp_growth = st.slider("Crescimento do PIB (%)", min_value=-10.0, max_value=10.0, value=2.0, step=0.1)
    fuel_price = st.number_input("Índice de Preço do Combustível", min_value=0.5, max_value=3.0, value=1.0, step=0.05)

if st.button("Prever Vendas", use_container_width=True):
    region_enc = le_region.transform([region])[0]
    model_enc = le_model.transform([model_name])[0]

    input_data = pd.DataFrame([[year, month, region_enc, model_enc, avg_price, bev_share, gdp_growth, fuel_price]],
                               columns=["Year", "Month", "Region_enc", "Model_enc", "Avg_Price_EUR", "BEV_Share", "GDP_Growth", "Fuel_Price_Index"])

    prediction = model.predict(input_data)[0]

    st.metric(label="Unidades Previstas", value=f"{int(prediction):,}".replace(",", "."))
    st.success(f"O modelo prevê aproximadamente **{int(prediction):,}** unidades vendidas para o {model_name} em {region} ({month:02d}/{year}).".replace(",", "."))
