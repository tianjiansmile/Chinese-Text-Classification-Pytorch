# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle

def check():
    # 词向量数据
    a = np.load('THUCNews/data/' + 'embedding_SougouNews.npz')["embeddings"].astype('float32')
    print(a)

    # 词表数据，词与词向量下标的字典
    # file=open("THUCNews/data/vocab.pkl","rb")
    # data=pickle.load(file)
    # print(len(data))
    # file.close()

    # 既然是文本分类，那么一句话就是一个样本，而不是一个单词，一句话就是若干个单词组成
    # 那么一条文本的向量表示就是一个矩阵，全量的文本样本的表示才是一个张量，所以训练的时候输入是一个张量
    # 相当于将一句话变成了一个图像，然后进入CNN训练，通过标签，通过反向传播算法训练，
    #

if __name__ == '__main__':
    check()
