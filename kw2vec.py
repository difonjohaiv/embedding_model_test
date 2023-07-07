from utils import get_embedding
from sklearn.metrics.pairwise import cosine_similarity
import torch
from tqdm import tqdm
import pandas as pd
import pickle
import numpy as np


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


def build_view(save_path, kw_embedding_list):
    kw_df = pd.DataFrame(columns=['start', 'end', 'scores'])
    for i in tqdm(range(len(kw_embedding_list))):
        for j in range(len(kw_embedding_list)):
            news_01 = kw_embedding_list[i]
            news_02 = kw_embedding_list[j]
            score = cosine_similarity(news_01, news_02)
            sum = np.sum(score[score > 0.6])
            if sum >= 1.8:
                # print(f"新闻{i}新闻{j}相似,起码有3个关键词相似")
                new_row_data = {'start': i, 'end': j, 'scores': sum}
                new_row = pd.DataFrame(new_row_data, index=[0])
                kw_df = pd.concat([kw_df, new_row], ignore_index=True)
            else:
                continue
            #     print(f"新闻{i}新闻{j}不相似,Nono!!!!!")
    kw_df.to_csv(save_path, sep=" ", header=False, index=False)


if __name__ == '__main__':
    save_path = "thucnews/views/keyword_views.csv"
    with open("thucnew/views/pkl/keyword_embeddings.pickle", "rb") as file:
        kw_embedding_list = pickle.load(file=file)
    build_view(save_path=save_path, kw_embedding_list=kw_embedding_list)
