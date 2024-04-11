# 读取数据
import pandas as pd

train_data = pd.read_csv("../data/step3_data/final_data/Train.csv")
dev_data = pd.read_csv("../data/step3_data/final_data/Dev.csv")
test_data = pd.read_csv("../data/step3_data/final_data/Test.csv")

# 映射标签到1到4的范围内
label_mapping = {"辱骂": 0, "歧视": 1, "广告骚扰": 2, "色情污秽": 3}

# 将标签列转换为整数形式
train_data["辱骂"] = train_data["辱骂"].map(label_mapping)
dev_data["辱骂"] = dev_data["辱骂"].map(label_mapping)
test_data["辱骂"] = test_data["辱骂"].map(label_mapping)

# 查看转换后的数据
train_data.to_csv("../data/step3_data/final_data/Train_new.csv", index=False)
dev_data.to_csv("../data/step3_data/final_data/Dev_new.csv", index=False)
test_data.to_csv("../data/step3_data/final_data/Test_new.csv", index=False)
