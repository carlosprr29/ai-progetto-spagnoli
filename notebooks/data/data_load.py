import pandas as pd

df = pd.read_parquet("data/train-00000-of-00001-290868f0a36350c5.parquet")
print(df.head())
