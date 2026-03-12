# BMW Sales Predictor

Aplicação de Machine Learning para prever unidades vendidas de modelos BMW com base em dados globais de 2018 a 2025.

## Como executar

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

2. Treine o modelo:
```bash
python train.py
```

3. Inicie a aplicação:
```bash
streamlit run app.py
```

## Estrutura

```
projeto_bmw/
├── train.py        # Treinamento do modelo
├── modelo.pkl      # Modelo salvo
├── le_region.pkl   # Encoder da região
├── le_model.pkl    # Encoder do modelo BMW
├── app.py          # Aplicação Streamlit
├── requirements.txt
└── dataset.csv
```

## Tecnologias

- Python 3.9+
- Streamlit
- Scikit-Learn (Random Forest Regressor)
- Joblib
- Pandas
