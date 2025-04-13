import pandas as pd

df = pd.read_csv("analyzed_results.csv", encoding="utf-8")
df.to_excel("analyzed_results.xlsx", index=False, engine="openpyxl")
