# Titanic Survival Predictor

Aplicação de Machine Learning para prever se um passageiro do Titanic sobreviveria ao naufrágio.

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
projeto_titanic/
├── train.py        # Treinamento do modelo
├── modelo.pkl      # Modelo salvo
├── app.py          # Aplicação Streamlit
├── requirements.txt
└── train.csv
```

## Tecnologias

- Python 3.9+
- Streamlit
- Scikit-Learn (Random Forest Classifier)
- Joblib
- Pandas
# RandomFlorest
