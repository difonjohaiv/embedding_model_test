from utils import get_embedding
import pandas as pd
import torch
import pickle
from tqdm import tqdm


fn = "thucnews/thucnews.csv"
df = pd.read_csv(fn)

embedding = get_embedding(model_name="m3e-base")

contents = df['content']

content_embedding_large = torch.zeros(len(contents), 768)

x = torch.zeros(768)

for i in tqdm(range(len(contents))):
    x = torch.tensor(embedding.embed_query(contents[i]), dtype=float)
    content_embedding_large[i, :] = x
    i = i + 1

with open("thucnews/m3e_base/pkl/content_embedding_m3e_base.pkl", "wb") as file:
    pickle.dump(obj=content_embedding_large, file=file)
