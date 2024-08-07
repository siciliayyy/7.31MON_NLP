# todo:据说可以统一编码格式但是为啥没效果？
import jieba
import pandas as pd
from gensim.models import Word2Vec
vector_size = 128

train_data = pd.read_csv('data/restaurant_comment/train.csv', encoding='utf-8', sep='\t')
# print(train_data)

test_data = pd.read_csv('data/restaurant_comment/test_new.csv', encoding='utf-8')
# print(test_data.head())

comment = pd.concat([train_data[['comment']], test_data[['comment']]], axis=0, ignore_index=True)  # todo?

print(comment)


def CutWord(sentence):
    word = sentence.map(lambda x: [w for w in list(jieba.cut(x)) if len(w) != 1])  # todo：停用词咋没了？
    return word


comment['comment'] = CutWord(comment['comment'])
print(comment)

