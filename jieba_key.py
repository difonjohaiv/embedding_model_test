import jieba.analyse
import pandas as pd
from tqdm import tqdm


# 使用jieba提取关键词，选取top_10，以字符串的形式保存
def extract_kw_jieba(k):
    fn = f"thucnews/all_news/news_train_{k}_entity.csv"
    save = f"thucnews/all_news/news_train_{k}_entity_keyword.csv"
    df = pd.read_csv(fn)
    df['keyword'] = None
    for i in tqdm(range(len(df))):
        content = df['content'][i]
        kw_list = jieba.analyse.textrank(content,
                                         topK=10,
                                         withWeight=False,
                                         allowPOS=('ns', 'n', 'vn', 'v'))
        if len(kw_list) < 10:
            df.drop(index=i, inplace=True)
            continue
        keywords = "#".join(kw_list)
        df['keyword'][i] = keywords
    df.to_csv(save, index=False)


if __name__ == '__main__':
    indexs = [1, 2, 3, 4, 5, 6]
    for i in indexs:
        extract_kw_jieba(i)
        print(f"正在处理第 {i} 个文件ing")
