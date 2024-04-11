import pandas as pd
import random

good_comments_df = pd.read_csv('../data/raw_data/zhengchang.csv', header=None, usecols=[0])
bad_comments_df = pd.read_csv('../data/raw_data/buliangpinglun.csv', header=None, usecols=[0])

bad_comments_list = bad_comments_df[0].tolist()
good_comments_list = good_comments_df[0].tolist()


# print(bad_comments_list)
def synonym_replacement(sentences, synonym_dict, n=1):
    """
    对句子列表进行同义词替换
    :param sentences: 包含句子的列表
    :param synonym_dict: 同义词词典，格式为{单词: [同义词1, 同义词2, ...]}
    :param n: 每个句子要替换的同义词的数量
    :return: 替换后的句子列表
    """
    augmented_sentences = []
    for sentence in sentences:
        print(sentence)
        words = sentence.split()
        new_words = words.copy()
        random_word_list = list(set([word for word in words]))
        random.shuffle(random_word_list)
        num_replaced = 0

        for random_word in random_word_list:
            if random_word in synonym_dict:
                synonyms = synonym_dict[random_word]
                if len(synonyms) >= 1:
                    synonym = random.choice(synonyms)
                    new_words = [synonym if word == random_word else word for word in new_words]
                    num_replaced += 1
            if num_replaced >= n:
                break

        augmented_sentence = ''.join(new_words)
        augmented_sentences.append(augmented_sentence)

    return augmented_sentences


def read_synonym_txt(file_path):
    """
    从TXT文件中读取同义词词典
    :param file_path: 文件路径
    :return: 同义词词典，格式为{单词: [同义词1, 同义词2, ...]}
    """
    synonym_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            words = line.strip().split(',')
            if len(words) >= 2:
                word = words[0]
                synonyms = words[2:]
                synonym_dict[word] = synonyms
    return synonym_dict


# 从TXT文件中读取同义词词典
synonym_dict = read_synonym_txt('../data/dict/同义关系库.txt')
# print(synonym_dict)
# 对句子列表进行同义词替换
bad_augmented_sentences = synonym_replacement(bad_comments_list, synonym_dict, n=1)
good_augmented_sentences = synonym_replacement(good_comments_list, synonym_dict, n=1)
# print(augmented_sentences)
bad_comments_list += bad_augmented_sentences
good_comments_list += good_augmented_sentences

# bad_labels = [1] * len(bad_augmented_sentences) + [0] * len(bad_comments_df)
bad_labels = 1
# good_labels = [1] * len(good_augmented_sentences) + [0] * len(good_comments_df)
good_labels = 0
# 创建新的 DataFrame 包括评论和标签
bad_data = pd.DataFrame({'comment': bad_comments_list, 'label': bad_labels})
good_data = pd.DataFrame({'comment': good_comments_list, 'label': good_labels})

# 将数据合并
combined_data = pd.concat([bad_data, good_data], ignore_index=True)

# 生成两个单独的 CSV 文件
combined_data.to_csv('data/final_data/combined_data.csv', index=False)

# 分别写入 bad 和 good 的 CSV 文件
bad_data.to_csv('data/final_data/bad_comments.csv', index=False)
good_data.to_csv('data/final_data/good_comments.csv', index=False)
