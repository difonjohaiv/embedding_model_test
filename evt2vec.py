import csv
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import pandas as pd


def build_view(save_path, kw_embedding_list, threshold, num):
    print("正在构建图ing")
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
    threshold = 0.6
    num_list = [2, 3, 5, 7]
    for num in num_list:
        save_path = f"thucnews/t2v_large/event_views_{threshold}_{num}_large.csv"
        with open("thucnews/t2v_large/pkl/event_embeddings_large.pickle", "rb") as file:
            kw_embedding_list = pickle.load(file=file)
        build_view(save_path=save_path, kw_embedding_list=kw_embedding_list, threshold=threshold, num=num)
        print("正在清理文件ing")
        df = pd.read_csv(save_path)
        df.to_csv(save_path, sep=' ', index=False, header=False)
