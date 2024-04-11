import csv
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


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

# 编码函数，将数据转化成bert可以理解的内部表示
def encoder(max_len, vocab_path, text_list):
    # 加载分词模型
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    tokenizer = tokenizer(text_list, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    # 返回的类型为pytorch tensor
    input_ids = tokenizer['input_ids']
    token_type_ids = tokenizer['token_type_ids']
    attention_mask = tokenizer['attention_mask']
    return input_ids, token_type_ids, attention_mask


def load_data(path):
    csvFileObj = open(path, 'r', encoding='utf-8')
    readerObj = csv.reader(csvFileObj)
    next(readerObj)  # 跳过第一行
    text_list = []
    labels = []
    for row in readerObj:
        # label在什么位置就改成对应的index
        label = int(row[2])
        text = row[1]
        text_list.append(text)
        labels.append(label)
    # 调用encoder函数，获得预训练模型的三种输入形式
    input_ids, token_type_ids, attention_mask = encoder(max_len=180,
                                                        vocab_path="../bert-base-chinese",
                                                        text_list=text_list)
    labels = torch.tensor(labels)
    # 将encoder的返回值以及label封装为Tensor的形式
    data = TensorDataset(input_ids, token_type_ids, attention_mask, labels)
    return data


# 设定batch_size  8 16 32 64  这个参数取决于机器的性能
batch_size = 8
# 数据路径
train_data_path = "../data/final_data/Train.csv"
dev_data_path = "../data/final_data/Dev.csv"
test_data_path = "../data/final_data/Test.csv"
# 调用load_data函数，将数据加载为Tensor形式
train_data = load_data(train_data_path)
dev_data = load_data(dev_data_path)
test_data = load_data(test_data_path)
# 查看train_data的长度是否与文档一致
# print("Total samples in train_data:", len(train_data))

# 将训练数据和测试数据进行DataLoader实例化
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dataset=dev_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)



def train_model(model, train_loader, dev_loader, num_epochs=2, lr=2e-5, print_every=1,
                save_path='model/bert_model.pth', log_path='log/bert_log.csv'):
    with open(log_path, 'w', newline='') as csvfile:
        fieldnames = ['Epoch', 'Train_Loss', 'Dev_Accuracy', 'Dev_Precision', 'Dev_Recall', 'Dev_F1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 将模型的参数和缓存移动到GPU上执行
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1, verbose=True)
        criterion = nn.CrossEntropyLoss()

        train_losses = []  # 用于存储每个epoch的平均训练损失
        dev_accuracies = []  # 用于存储每个epoch的验证集准确率
        dev_precisions = []  # 用于存储每个epoch的验证集精确率
        dev_recalls = []  # 用于存储每个epoch的验证集召回率
        dev_f1_scores = []  # 用于存储每个epoch的验证集F1分数

        best_f1_score = 0.0

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for step, (input_ids, token_type_ids, attention_mask, labels) in tqdm(enumerate(train_loader),
                                                                                  desc=f'Epoch {epoch + 1}/{num_epochs} -'
                                                                                       f'Training'):
                input_ids, token_type_ids, attention_mask, labels = input_ids.to(device), token_type_ids.to(
                    device), attention_mask.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(input_ids, token_type_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if step % print_every == 0:
                    print(f'Batch {step}/{len(train_loader)}, Loss: {loss.item():.4f}')

            average_loss = total_loss / len(train_loader)
            train_losses.append(average_loss)

            model.eval()
            all_labels = []
            all_predicted = []
            with torch.no_grad():
                for step, (input_ids, token_type_ids, attention_mask, labels) in tqdm(enumerate(dev_loader),
                                                                                      desc=f'Epoch {epoch + 1}/{num_epochs}'
                                                                                           f' - Validation'):
                    input_ids, token_type_ids, attention_mask, labels = input_ids.to(device), token_type_ids.to(
                        device), attention_mask.to(device), labels.to(device)

                    outputs = model(input_ids, token_type_ids, attention_mask)
                    _, predicted = torch.max(outputs, 1)

                    all_labels.extend(labels.cpu().numpy())
                    all_predicted.extend(predicted.cpu().numpy())

                accuracy = accuracy_score(all_labels, all_predicted)
                recall = recall_score(all_labels, all_predicted)
                precision = precision_score(all_labels, all_predicted)
                f1 = f1_score(all_labels, all_predicted)

                dev_accuracies.append(accuracy)
                dev_precisions.append(precision)
                dev_recalls.append(recall)
                dev_f1_scores.append(f1)

                print(
                    f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}, Validation Accuracy: {accuracy:.4f}'
                    f', Validation Precision: {precision:.4f}, Validation Recall: {recall:.4f}, Validation F1 Score: {f1:.4f}')

                scheduler.step(average_loss)

                if f1 > best_f1_score:
                    best_f1_score = f1
                    torch.save(model.state_dict(), save_path)
                    print(f'Model saved with the best F1 score: {best_f1_score}')

                writer.writerow({'Epoch': epoch + 1, 'Train_Loss': average_loss, 'Dev_Accuracy': accuracy,
                                 'Dev_Precision': precision, 'Dev_Recall': recall, 'Dev_F1': f1})

    plot_learning_curve(train_losses, dev_accuracies, dev_precisions, dev_recalls, dev_f1_scores)


def plot_learning_curve(train_losses, dev_accuracies, dev_precisions, dev_recalls, dev_f1_scores):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss')

    plt.subplot(2, 1, 2)
    plt.plot(dev_accuracies, label='Validation Accuracy')
    plt.plot(dev_precisions, label='Validation Precision')
    plt.plot(dev_recalls, label='Validation Recall')
    plt.plot(dev_f1_scores, label='Validation F1 Score')
    plt.title('Validation Metrics')
    plt.legend()

    plt.show()


# 创建模型
output_dim = 2  # 有两类情感
model = BERTClassificationModel(output_dim=output_dim)

# 训练模型并保存
train_model(model, train_loader, dev_loader, num_epochs=5, lr=2e-5, print_every=1, save_path='../model/bert_model.pth',
            log_path='../log/bert_log.csv')

