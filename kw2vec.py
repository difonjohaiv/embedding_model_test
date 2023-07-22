from utils import get_embedding
from sklearn.metrics.pairwise import cosine_similarity
import torch
from tqdm import tqdm
# import pandas as pd
import pickle
import numpy as np
import csv


# 分割成10个kw
def get_each_keyword(x):
    x_list = x.split("#")
    return x_list


# 分别对每个新闻选取出来的10个关键字进行嵌入编码
embedding = get_embedding()


def embed_keyword(x):
    x_embeddings = torch.zeros(10, 768)
    i = 0
    for item in x:
        kw_embedding = torch.tensor(embedding.embed_query(item), dtype=float)
        x_embeddings[i, :] = kw_embedding
        i = i + 1
    return x_embeddings


def embed_corpus(kws):
    # 把每个新闻的关键字嵌入保存到列表中
    with torch.no_grad():
        kw_embedding_list = []
        for item in tqdm(kws):
            kw_list = get_each_keyword(item)
            kw_embedding_list.append(embed_keyword(kw_list))
        print(len(kw_embedding_list))


def build_view(save_path, kw_embedding_list, threshold, num):
    with open(save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        writer.writerow(['Row', 'Column', 'Value'])
        for i in tqdm(range(len(kw_embedding_list))):
            for j in range(len(kw_embedding_list)):
                news_01 = kw_embedding_list[i]
                news_02 = kw_embedding_list[j]
                score = cosine_similarity(news_01, news_02)
                sum = np.sum(score[score >= threshold])
                if sum >= (threshold * num):
                    writer.writerow([i, j, sum])
                else:
                    continue


if __name__ == '__main__':
    threshold = 0.7
    num = 2
    save_path = f"thucnews/views/keyword_views_{threshold}_{num}.csv"
    print(save_path)
    with open("thucnews/views/pkl/keyword_embeddings.pickle", "rb") as file:
        kw_embedding_list = pickle.load(file=file)
    build_view(save_path=save_path, kw_embedding_list=kw_embedding_list, threshold=threshold, num=num)
