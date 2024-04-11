import jieba
import pickle
from torch import nn
from transformers import BertTokenizer, BertModel
import torch
import matplotlib.pyplot as plt

class BERTClassificationModel(nn.Module):
    def __init__(self, output_dim, pretrained_name='../bert-base-chinese', weight_decay=0.01):
        super(BERTClassificationModel, self).__init__()
        # 定义 Bert 模型
        self.bert = BertModel.from_pretrained(pretrained_name)
        for param in self.bert.parameters():
            param.requires_grad = True
        # 定义外接的一个全连接层
        self.dense = nn.Linear(768, output_dim)
        self.weight_decay = weight_decay  # 设置正则化参数

    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[1]
        linear_output = self.dense(bert_cls_hidden_state)
        return linear_output

    def optimizer_parameters(self):
        # 返回需要进行优化的参数以及对应的权重衰减（正则化）系数
        params = self.parameters()
        decay = self.weight_decay
        if decay > 0.0:
            return [{'params': params, 'weight_decay': decay}]
        else:
            return [{'params': params}]


# Bert_biLSTM模型
class BERTBiLSTMClassificationModel(nn.Module):
    def __init__(self, output_dim, pretrained_name='../bert-base-chinese', lstm_hidden_size=256, lstm_num_layers=1):
        super(BERTBiLSTMClassificationModel, self).__init__()

        # 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained(pretrained_name)
        for param in self.bert.parameters():
            param.requires_grad = True  # 可以冻结BERT模型的参数，避免在微调过程中更新

        # 添加BiLSTM层
        self.lstm = nn.LSTM(input_size=768, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers,
                            batch_first=True, bidirectional=True)

        # 添加全连接层
        self.fc = nn.Linear(lstm_hidden_size * 2, output_dim)  # 注意乘以2，因为BiLSTM是双向的

    def forward(self, input_ids, token_type_ids, attention_mask):
        # 获取BERT模型输出
        bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[1]  # 取得CLS位置的隐藏状态

        # 将BERT的输出传递给BiLSTM层
        lstm_output, _ = self.lstm(bert_cls_hidden_state.unsqueeze(1))  # 添加一维，适应LSTM输入格式

        # 将BiLSTM的输出传递给全连接层
        output = self.fc(lstm_output.squeeze(1))  # 去除添加的维度

        return output

def load_sensitive_words_dict(file_path):
    """
    从文本文件中读取敏感词典
    :param file_path: 敏感词典文件路径
    :return: 敏感词列表
    """
    sensitive_words = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            word = line.strip()  # 去除换行符和空白字符
            sensitive_words.append(word)
    return sensitive_words


# 判断是否包含敏感词
def contains_sensitive_words(text, sensitive_words):
    words = jieba.cut(text)
    word_list = list(words)
    for word in sensitive_words:
        if word in word_list:
            return True
    return False


# 判别是否为不良评论预测
def predict_sentiment(text, model_path='../model/bert_bilstm_model.pth', max_length=200):
    # 初始化BERTBiLSTMClassificationModel模型
    output_dim = 2  # 替换成你的输出维度
    model = BERTBiLSTMClassificationModel(output_dim=output_dim)

    # 加载已经训练好的权重
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained('../bert-base-chinese')

    # 将文本进行tokenization
    inputs = tokenizer(text, max_length=max_length, return_tensors='pt', truncation=True)

    # 获取输入tensor
    input_ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']
    attention_mask = inputs['attention_mask']

    # 使用模型进行预测
    with torch.no_grad():
        output = model(input_ids, token_type_ids, attention_mask)

    # 处理模型输出
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

    # 根据实际需求，你可以返回概率分布、预测的类别，或者其他信息
    return predicted_class

def predict_sentiment2(text, model_path='../model/2bert_model.pth', max_length=200):
    # 初始化BERTBiLSTMClassificationModel模型
    output_dim = 4  # 替换成你的输出维度
    model = BERTClassificationModel(output_dim=output_dim)

    # 加载已经训练好的权重
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained('../bert-base-chinese')

    # 将文本进行tokenization
    inputs = tokenizer(text, max_length=max_length, return_tensors='pt', truncation=True)

    # 获取输入tensor
    input_ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']
    attention_mask = inputs['attention_mask']

    # 使用模型进行预测
    with torch.no_grad():
        output = model(input_ids, token_type_ids, attention_mask)

    # 处理模型输出
    probabilities = torch.softmax(output, dim=1)

    # 返回概率分布
    return probabilities


# 主函数
def main(text):
    # 加载敏感词典
    sensitive_words = load_sensitive_words_dict('../data/dict/Sensitive_words.txt')

    # 判断是否包含敏感词
    if contains_sensitive_words(text, sensitive_words):
        print("包含敏感词，请管理人员审核")
    else:
        predicted_class = predict_sentiment(text, model_path='../model/bert_bilstm_model.pth', max_length=200)
        # print(predicted_class)
        if predicted_class == 0:
            print("正常评论")
        else:
            output_probs = predict_sentiment2(text, model_path='../model/2bert_model.pth', max_length=200)
            label_to_text = {0: "辱骂", 1: "歧视", 2: "广告骚扰", 3: "色情污秽"}
            predicted_label = output_probs.argmax().item()
            predicted_text = label_to_text[predicted_label]
            print("涉嫌不良评论:", predicted_text)

            # 输出每个类型的概率
            probabilities = [prob.item() for prob in output_probs.squeeze()]
            for label, prob in enumerate(probabilities):
                print(f"{label_to_text[label]}: {prob:.2%}")

            # 创建柱状图
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
            plt.figure(figsize=(8, 6))
            colors = ['skyblue'] * len(label_to_text)
            max_index = probabilities.index(max(probabilities))
            colors[max_index] = '#c81623'
            plt.bar(label_to_text.values(), probabilities, color=colors)
            plt.xlabel('评论类型')
            plt.ylabel('概率')
            plt.title('不良评论类型概率分布')
            plt.xticks(rotation=45)
            plt.ylim(0, 1)  # 设置纵轴范围为 0 到 1
            plt.show()


# 测试
text = "这群东北人是不是都这样，没脑子"
main(text)
