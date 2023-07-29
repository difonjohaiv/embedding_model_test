from utils import get_embedding
import pandas as pd
import torch
import pickle
from tqdm import tqdm


fn = "thucnews/thucnews.csv"
df = pd.read_csv(fn)

embedding = get_embedding()

contents = df['content']

content_embedding_large = torch.zeros(len(contents), 1024)

x = torch.zeros(1024)

for i in tqdm(range(len(contents))):
    x = torch.tensor(embedding.embed_query(contents[i]), dtype=float)
    content_embedding_large[i, :] = x
    i = i + 1

with open("thucnews/t2v_large/pkl/content_embedding_large.pkl", "wb") as file:
    pickle.dump(obj=content_embedding_large, file=file)
