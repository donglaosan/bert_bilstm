import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# 加载保存的模型
with open('nb_classifier_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

text_to_predict = "你妈的，我看你像坨屎"  # 替换成你的待预测文本

with open('nb_classifier_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# 使用与训练时相同的文本向量化方法（TF-IDF）
X_text = vectorizer.transform([text_to_predict])

# 使用加载的模型进行预测
prediction = loaded_model.predict(X_text)


# 打印预测结果
print("预测结果:", prediction)
