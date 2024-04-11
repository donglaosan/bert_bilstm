import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 读取数据集（假设数据集中有两列，一列是文本内容，一列是情感标签）
data = pd.read_csv('../../data/final_data/combined_data.csv')

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data['comment'], data['label'], test_size=0.2, random_state=42)

# 使用 TF-IDF 进行特征提取
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

with open('nb_classifier_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)


# 训练朴素贝叶斯分类器
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

with open('nb_classifier_model.pkl', 'wb') as f:
    pickle.dump(nb_classifier, f)

with open('nb_classifier_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
# 使用训练好的模型进行预测
y_pred = nb_classifier.predict(X_test_tfidf)

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 打印评估指标
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
