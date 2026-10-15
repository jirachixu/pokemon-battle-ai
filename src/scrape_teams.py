import pandas as pd
import requests

df = pd.read_csv("./championsmcpastes.csv", skiprows=2)

for index, row in df.iterrows():
    event = str(row["Tournament / Event"]).lower()
    if ("regional" in event or "international" in event or "worlds" in event) and row["EVs"] == "Yes":
        response = requests.get(f"{row["Pokepaste"]}/raw")
        with open(f"./teams/{row["Team ID"]}.txt", "w") as f:
            f.write(response.text)