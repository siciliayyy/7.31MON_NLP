import re

import jieba
import numpy as np
import pandas as pd

path = 'data/preprocess/preprocess_1.csv'


def filter_it(data):
    data = re.sub('[，。！？\s\*…<>~&、～\\/]', '', data)
    data = re.sub('[\[\]@#$%\^\&_\+-]', '', data)

    data = re.sub('#.*?#', '', data)
    data = re.sub('（.*?）', '', data)
    data = re.sub('【.*?】', '', data)
    data = re.sub('《。*？》', '', data)
    return data


def cutwords(line, stopwords):  # 分词后删除停用词
    line = list(jieba.cut(line))
    for word in line:
        if word in stopwords:
            line.remove(word)
    return line


# 读取训练集
train_file = pd.read_csv('data/restaurant_comment/train.csv', sep='\t')
test_file = pd.read_csv('data/restaurant_comment/test_new.csv', sep=',')

# 注意注意这里的sep = '\t'，因为标号和评论是写在一个单元格里的如果不用这个的话会分不开。
# 读取label
train_mask_load = np.array(train_file['label'].tolist())

# 逐行遍历去除符号
train_filter = train_file['comment'].apply(lambda x: filter_it(x))  # df.apply为逐行遍历，是最方便的一种
test_filter = test_file['comment'].apply(lambda x: filter_it(x))

# 保存dataframe为csv
train_filter.to_csv('data/preprocess/preprocess_1.csv', index=True, header=False)

# 读入停用词表
stopwords = list()
with open('data/stopwords/哈工大停用词表.txt', encoding='utf-8') as f:  # todo: 之前一直报错就是没用utf-8
    for line in f.read().split('\n'):  # todo:记得去掉结尾的\n
        stopwords.append(line)

# 分词并删除停用词   呈[]
train_aftercut = [cutwords(sentence, stopwords) for sentence in train_filter]  # 将dataframe逐行处理完读到列表中
test_aftercut = [cutwords(sentence, stopwords) for sentence in test_filter]

# ['一如既往 好吃 希望 开 其他 城市', '味道 很 不错 分量 足 客人 很多 满意']  这样才能用搞成BOW
train_list = list()
for line in train_aftercut:
    train_list.append(' '.join(line))

test_list = list()
for line in test_aftercut:
    test_list.append(' '.join(line))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


'''
用TF-IDF生成句向量
'''
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(train_list + test_list)
train_input_load = tfidf_vectorizer.transform(train_list).toarray()
test_input = tfidf_vectorizer.transform(test_list).toarray()



'''
用BOW生成句向量
'''
# todo:这里注意了，fit要将train和test一起(要读入train和test的的所有词不然预测test时维数不一样)，transform时要分开
# vectorizer = CountVectorizer()
# vectorizer.fit(train_list + test_list)
# train_input_load = vectorizer.transform(train_list).toarray()
# test_input = vectorizer.transform(test_list).toarray()

# todo:这里将train和test分开进行BOW操作，结果train和 test的句子向量维数不一致，结果无法对test 做出预测wrong
# count = vectorizer.fit_transform(train_list)
# train_input_load = count.toarray()  # train_input 还要分成train和validation所以这里用load来表示
#
# count = vectorizer.fit_transform(test_list)
# test_input = count.toarray()  # test_input 直接放入模型中跑因此不用分出load.

# 逐行遍历，看是否有全部为0的行_结果没有！
# for i in range(train_input.shape[0]):
#     print(train_input[i, :].sum())


from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019).split(train_input_load, train_mask_load)
test_preds = list()
test_preds = np.array(test_preds)
for i, (train_idx, valid_idx) in enumerate(skf):
    train_input, train_mask = train_input_load[train_idx], train_mask_load[train_idx]
    valid_input, valid_mask = train_input_load[valid_idx], train_mask_load[valid_idx]

    lr = LogisticRegression(C=1.2)
    lr.fit(train_input, train_mask)

    valid_preds = lr.predict(valid_input)
    test_preds = lr.predict_proba(test_input)   # todo:每个据向量的维数为2，即为在某个下标下对应的概率

    acc = accuracy_score(valid_mask, valid_preds)
    print(acc)
    f1 = f1_score(valid_mask, valid_preds, average='macro')

test_mask = [np.argmax(r) for r in test_preds]  # todo:返回每一行最大的下标
sub = test_file.copy()
sub['label'] = test_mask
sub[['id', 'comment', 'label']].to_csv('data/preprocess/preprocess_1.csv', index=None)
