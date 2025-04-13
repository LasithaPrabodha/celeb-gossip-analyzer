import pandas as pd

df = pd.read_csv("./output/analyzed_results.csv", encoding="utf-8")
df.to_excel("./output/analyzed_results.xlsx", index=False, engine="openpyxl")
