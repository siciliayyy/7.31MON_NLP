import jieba
import pandas as pd

# todo: load files
# file = pd.read_csv('data/train.csv', sep='\t', encoding='UTF-8')  # 读取为csv文件
# with open('data/train/train.txt', encoding='UTF-8') as f:  # 必须加上UTF-8 不然识别不了中文报错
#     for line in f.read().split('\n'):
#         print(line)


# todo:jieba 初步使用
s = '下雨天来的，没有想象中那么火爆。环境非常干净，古色古香的，我自己也是个做服务行业的，我都觉得他们的服务非常好，场地脏了马上就有阿姨打扫。'

aaa = jieba.cut(s, cut_all=True)  # 返回Generator生成器类型   全模式，所有出现的词语都会录入
print("\n" + " ".join(aaa))  # todo : join:iterable->str

bbb = jieba.cut(s, cut_all=False)  # 精确模式
print(" ".join(bbb))

bbb = jieba.cut(s)  # 默认模式
print(" ".join(bbb))

jieba.add_word('脏了')

bbb = jieba.cut(s, cut_all=False)  # 增加语料库
print('添加词语后:  ' + " ".join(bbb))

# todo: TF-IDF 关键词抽取
from jieba import analyse

TF_IDF = analyse.extract_tags

sentence = '线程是程序执行时的最小单位，它是进程的一个执行流，是 CPU 调度和分派的基本单位，一个进程可以由很多个线程组成，' \
           '线程间共享进程的所有资源，每个线程有自己的堆栈和局部变量。' \
           '线程由 CPU 独立调度执行，在多 CPU 环境下就允许多个线程同时运行。' \
           '同样多线程也可以实现并发操作，每个请求分配一个线程来处理。'
print()
aaa = TF_IDF(sentence, withWeight=True)
print(aaa)

# todo:word2vec
from gensim.models import Word2Vec

sentence = '凉皮有味道了，吃完一天肚子都不舒服，拉肚子，不是第一次在这家吃了，希望有所改进' \
           '帅哥经理又帅，服务又好，以后会经常光顾他们家，为了帅哥经理去的哦，挺好的，挺棒的，非常好，不错的'
words = jieba.cut(sentence)
seperated_word = list()
ss = list()
for word in words:
    seperated_word.append(word)
ss.append(seperated_word)
# 训练模型    第一个参数sentence须是一个列表(像这种：[['aaa', 'aa', 'aa']]),min_count为最小出现频率数
mod = Word2Vec(ss, sg=1, min_count=1)
# 保存模型
mod.wv.save_word2vec_format('result/models/word2vec/word2vec.txt')  # todo:可保存为txt
# 读取模型
mod = mod.wv.load_word2vec_format('result/models/word2vec/word2vec.txt')
# print(mod['经理'])
print()
print(mod.most_similar(positive=['好'], topn=10))
print(mod.similarity('好', '不错'))

# todo:cbow
import re
train = pd.read_csv('data/train.csv', sep='\t')
print(train)
text = re.sub('[，]', '', train['comment'])
print(text)
# test = pd.read_csv('data/test_new.csv')  # 为啥，不能删除
# print(test)



