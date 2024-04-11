import torch
from torch import nn
from transformers import BertTokenizer, BertModel


class BERTClassificationModel(nn.Module):
    def __init__(self, output_dim, pretrained_name='./bert-base-chinese', weight_decay=0.01):
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


def predict_sentiment(text, model_path='model/bert_model.pth', max_length=200):
    # 初始化BERTBiLSTMClassificationModel模型
    output_dim = 2  # 替换成你的输出维度
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
    predicted_class = torch.argmax(probabilities, dim=1).item()

    # 根据实际需求，你可以返回概率分布、预测的类别，或者其他信息
    return {predicted_class}


text_input = '你妈的，我看你像坨屎。'
# 调用模型进行预测
prediction = predict_sentiment(text_input)
print(prediction)
