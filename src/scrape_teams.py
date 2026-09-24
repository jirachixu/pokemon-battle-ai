import pandas as pd
import requests

df = pd.read_csv("./championsmcpastes.csv")

for index, row in df.iterrows():
    if ("top" in row["Team Description"].lower() \
        or "champion" in row["Team Description"].lower() \
        or "regional" in row["Team Description"].lower() \
        or "worlds" in row["Team Description"].lower()) \
        and row["EVs"] == "Yes":

        response = requests.get(f"{row["Pokepaste"]}/raw")
        with open(f"./teams/{row["Team ID"]}.txt", "w") as f:
            f.write(response.text)