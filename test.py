#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
==== No Bugs in code, just some Random Unexpected FEATURES ====
┌─────────────────────────────────────────────────────────────┐
│┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐│
││Esc│!1 │@2 │#3 │ 4 │%5 │^6 │&7 │*8 │(9 │)0 │_- │+= │|  │`~ ││
│├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴───┤│
││ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{[ │}] │ BS  ││
│├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤│
││ Ctrl │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│ ' │ Enter  ││
│├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────┬───┤│
││ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│Shift │Fn ││
│└─────┬──┴┬──┴──┬┴───┴───┴───┴───┴───┴──┬┴───┴┬──┴┬─────┴───┘│
│      │Fn │ Alt │         Space         │ Alt │Win│   HHKB   │
│      └───┴─────┴───────────────────────┴─────┴───┘          │
└─────────────────────────────────────────────────────────────┘
@File    :   test.py
@Time    :   2023/06/08 11:01:18
@Author  :   Joven Chu
@github  :   https://github.com/JovenChu
@Desc    :   对多个embediing模型进行词向量语义检测
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import random
import torch.cuda
import torch.backends
from sklearn.preprocessing import MinMaxScaler
# import os
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embedding_model_dict = {
    # "ernie-base": "models/ernie-3.0-base-zh",
    "bge-large": "models/bge-large-zh",
    "m3e-base": "models/m3e-base",
    "text2vec-base": "models/text2vec-base-chinese",
    "text2vec-large": "models/text2vec-large-chinese"
    # "sentence-transformers-v2": "models/sentence-transformers-v2"
}

EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"


def random_samples(sample_nums):
    """对每个类别的样本随机抽样100个,写入新的文件中
    """
    label_dict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}

    with open('data/Chinese-STS-B/sts-b-train.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items = line.split('\t')
            label = int(items[-1])
            label_dict[label].append(line)

    with open('data/sampled_data.txt', 'w', encoding='utf-8') as f:
        for label in label_dict:
            samples = random.sample(label_dict[label], sample_nums)
            for sample in samples:
                f.write(sample + '\n')


def get_x_linspace_y(label_dict, sample_nums):
    """根据样本的类别数量和样本数量生成等差数列列表
    """
    x_list = []
    y_list = []
    for l in label_dict:
        y_list += label_dict[l]
        x_nums = len(label_dict[l])
        xx = np.linspace(l, l + 1, x_nums)
        x_list += xx.tolist()

    # 将 x_list 归一化到 [0, 1] 区间
    x_array = np.array(x_list).reshape(-1, 1)
    scaler = MinMaxScaler()
    x_normalized = scaler.fit_transform(x_array)

    # x = np.array(x_list)
    y = np.array(y_list)
    return x_normalized, y


def model_test(embedding_model, sample_nums):
    """载入数据，分别使用模型计算相似度，得到相似度y的列表
    """
    label_dict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    # 加载模型

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_dict[embedding_model],
        model_kwargs={'device': EMBEDDING_DEVICE})

    # 加载数据进行预测y
    print("加载数据进行预测y......")
    result_all = ""
    with open('data/sampled_data.txt', 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.strip()
            # print(line)
            if not line:
                continue
            items = line.split('\t')
            label = int(items[-1])
            # 获取每个文本的向量
            vec1 = np.array(embeddings.embed_query(items[0]))
            vec2 = np.array(embeddings.embed_query(items[1]))
            # print(embed_list)
            # 计算文本间的相似度
            cos_sim = cosine_similarity(vec1.reshape(1, -1),
                                        vec2.reshape(1, -1))
            # print(cos_sim[0][0])
            # 记录y值
            label_dict[label].append(cos_sim[0][0])
            # 记录预测值
            result_all += line + '\t' + str(cos_sim[0][0]) + '\n'

    # 保存预测值记录
    file_path_r = 'result/txt/' + embedding_model + '_result.txt'
    with open(file_path_r, 'w', encoding='utf-8') as f:
        f.write(result_all)

    print("画图中......")
    # 画图
    # 使用linspace函数生成等差数列，将y点平均分布在x轴的特定区间中
    x, y = get_x_linspace_y(label_dict, sample_nums)
    # 画图
    # fname 为 你下载的字体库路径，注意 SourceHanSansSC-Bold.otf 字体的路径
    zhfont1 = matplotlib.font_manager.FontProperties(
        fname="font/SourceHanSansSC-Bold.otf")
    plt.figure(figsize=(20, 10))  # 画布大小
    plt.grid(ls='--')  # 设置虚线网格
    plt.scatter(x, y)
    plt.title("使用向量模型:  " + embedding_model, fontproperties=zhfont1)
    plt.ylabel("余弦相似度", fontproperties=zhfont1)
    plt.xlabel("样本标签区间(人工为数据标注的真实相似度分数,由[0-5]归一化到[0-1])",
               fontproperties=zhfont1)
    # plt.show()
    # 保存成文件
    plt.savefig('result/jpg/' + embedding_model + '_result.jpg')
    print("完成测试")
    return x, y


if __name__ == '__main__':
    # 抽样数量
    sample_nums = 200
    random_samples(sample_nums)

    # 创建一个画布
    plt.figure(figsize=(80, 40))
    # fname 为 你下载的字体库路径，注意 SourceHanSansSC-Bold.otf 字体的路径
    zhfont1 = matplotlib.font_manager.FontProperties(
        fname="font/SourceHanSansSC-Bold.otf")

    all_x = []
    all_y = []
    all_model = []

    # 全部模型运行测试
    for i, embedding_model in enumerate(embedding_model_dict):
        print("正在使用 【{}】 模型测试中。。。。。。".format(embedding_model))
        # 执行模型测试
        x, y = model_test(embedding_model, sample_nums)
        # 保存每个模型的结果
        all_x.append(x)
        all_y.append(y)
        all_model.append(embedding_model)

    # 绘制每个模型的结果，每个模型以不同颜色绘制
    for i, (x, y, embedding_model) in enumerate(zip(all_x, all_y, all_model)):
        plt.scatter(x, y, label=embedding_model)

    # 添加图例
    plt.legend()

    # 添加标题和轴标签
    plt.title("不同模型的余弦相似度比较", fontproperties=zhfont1)
    plt.ylabel("余弦相似度", fontproperties=zhfont1)
    plt.xlabel("样本标签区间(人工为数据标注的真实相似度分数,由[0-5]归一化到[0-1])",
               fontproperties=zhfont1)

    # 保存整个图
    plt.savefig('result/jpg/all_results.jpg')
    # 单独模型运行测试
    # embedding_model= "sentence-transformers-v2"
    # model_test(embedding_model, sample_nums)
