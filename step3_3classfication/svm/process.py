import pandas as pd

data = pd.read_csv("../../data/step3_data/buliangpinglun.csv")
label_mapping = {"辱骂": 0, "歧视": 1, "广告骚扰": 2, "色情污秽": 3}

data["辱骂"] = data["辱骂"].map(label_mapping)
data.to_csv("../../data/step3_data/final_data/buliangpinglun.csv", index=False)