import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("dataset.csv")

le_region = LabelEncoder()
le_model = LabelEncoder()

df["Region_enc"] = le_region.fit_transform(df["Region"])
df["Model_enc"] = le_model.fit_transform(df["Model"])

joblib.dump(le_region, "le_region.pkl")
joblib.dump(le_model, "le_model.pkl")

features = ["Year", "Month", "Region_enc", "Model_enc", "Avg_Price_EUR", "BEV_Share", "GDP_Growth", "Fuel_Price_Index"]
X = df[features]
y = df["Units_Sold"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "modelo.pkl")
print("Modelo treinado e salvo com sucesso!")
