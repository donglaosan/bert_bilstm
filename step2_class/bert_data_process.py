import pandas as pd
from sklearn.model_selection import train_test_split

# 读取最原始数据集
data = pd.read_csv("../data/final_data/combined_data.csv")
# 打乱数据集
shuffled_data = data.sample(frac=1, random_state=42).reset_index(drop=True)
# frac=1 表示使用全部数据，random_state=42用于确保结果的可重复性,reset_index(drop=True)重置索引.
shuffled_data.to_csv("data/mid_data/combined_data_plus.csv", index=True)
print("打乱完成")


def remove_spaces_from_csv(file_path):
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        # 删除每个字段中的空格
        df = df.applymap(lambda x: x.replace(" ", '') if isinstance(x, str) else x)
        # 通过正则表达式将被@的人尽量删掉
        pattern = f'@[\\w\\d\\s!@#$%^&*()_+-=:;\'",.<>?/\\\\|`~\\u4e00-\\u9fa5]*[： :]'
        df = df.replace(to_replace=pattern, value='', regex=True)
        # 通过正则表达式删除不需要的网址信息
        pattern2 = r'https?:\/\/[^\s]*'
        df = df.replace(to_replace=pattern2, value='', regex=True)
        # 保存修改后的数据框到原始CSV文件(删掉了第一行)
        df.to_csv(file_path, index=False, header=False, mode='w')
        print(f"空格已成功删除，被@的人已删除")
    except Exception as e:
        print(f"发生错误: {e}")


remove_spaces_from_csv("../data/mid_data/combined_data_plus.csv")


Data_path = "../data/mid_data/combined_data_plus.csv"
mid_data = pd.read_csv(Data_path,  encoding='utf-8')

train_dataset, temp_data = train_test_split(mid_data, test_size=0.2)
validate_dataset, test_dataset = train_test_split(temp_data, test_size=0.5)
print(len(train_dataset))
print(len(validate_dataset))
print(len(test_dataset))

# 设置保存路径
train_data_path = "../data/final_data/Train.csv"
dev_data_path = "../data/final_data/Dev.csv"
test_data_path = "../data/final_data/Test.csv"

# index参数设置为False表示不保存行索引,header设置为False表示不保存列索引
train_dataset.to_csv(train_data_path, index=False, header=True)
validate_dataset.to_csv(dev_data_path, index=False, header=True)
test_dataset.to_csv(test_data_path, index=False, header=True)
