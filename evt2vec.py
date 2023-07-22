import csv
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle


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
    threshold = 0.5
    num = 3
    save_path = f"thucnews/views/event_views_{threshold}_{num}.csv"
    with open("thucnews/views/pkl/event_embeddings.pickle", "rb") as file:
        kw_embedding_list = pickle.load(file=file)
    build_view(save_path=save_path, kw_embedding_list=kw_embedding_list, threshold=threshold, num=num)
