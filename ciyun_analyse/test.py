import pandas as pd
import jieba
import wordcloud

# 加载 CSV 文件
df = pd.read_csv('../data/step3_data/final_data/buliangpinglun.csv')

# 合并文本数据
text = ' '.join(df['comment'])

# 使用 jieba 进行分词
word_list = jieba.lcut(text)
text_processed = ' '.join(word_list)

# 生成词云图
wc = wordcloud.WordCloud(
    width=200,
    height=150,
    background_color='white',
    font_path='msyh.ttc',
    scale=20,
    stopwords={' ', '的', '我', '都', '不', '是', '梅西', '就是', '你', 'http', '这个', '被', '今天', '一个', '了', '但是', '觉得', '有点', '虽然', '我们','就','在','人','他','多','这','吗','啊'}
)
wc.generate(text_processed)

# 保存词云图
wc.to_file('wordcloud.png')
