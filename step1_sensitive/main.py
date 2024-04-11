import jieba


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


# 读取敏感词典文件
sensitive_words = load_sensitive_words_dict('../data/dict/Sensitive_words.txt')


def contains_sensitive_word(sentence, sensitive_words):
    """
    判断句子是否包含敏感词
    :param sentence: 待检测的句子
    :param sensitive_words: 敏感词典
    :return: True（句子包含敏感词）或 False（句子不包含敏感词）
    """
    # 分词
    words = jieba.cut(sentence)

    word_list = list(words)
    # 遍历敏感词典
    for word in sensitive_words:
        if word in word_list:
            return True
    return False


# 待检测的句子
sentence = "我去tmd"

# 判断句子是否包含敏感词
if contains_sensitive_word(sentence, sensitive_words):
    print("句子包含敏感词。")
else:
    print("句子不包含敏感词。")
