import pandas as pd
import joblib

data_path = "/content/drive/MyDrive/ML_Project/data.csv"
model_path = "/content/drive/MyDrive/ML_Project/model.pkl"

df = pd.read_csv(data_path)
model = joblib.load(model_path)

display(df.head())
print("Model and data loaded successfully")
