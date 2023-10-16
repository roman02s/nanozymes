import pandas as pd

# Заглушка
df = pd.read_csv("data/nanozymes_extended.csv")

def get_parameters(link: str) -> dict:
    print(df[df["link"] == link])
    return df[df["link"] == link].iloc[0].to_dict()
