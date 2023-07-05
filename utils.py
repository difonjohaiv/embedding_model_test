import torch
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from torch.nn import CosineSimilarity

EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"
model_kwargs = {'device': EMBEDDING_DEVICE}

embedding_model_dict = {
    "ernie-base": "models/ernie-3.0-base-zh",
    "ernie-xbase": "models/ernie-3.0-xbase-zh",
    "text2vec-base": "models/text2vec-base-chinese",
    "text2vec-large": "models/text2vec-large-chinese"
}


# 获取嵌入模型
def get_embedding(model_name="text2vec-base"):
    embedding = HuggingFaceEmbeddings(
        model_name=embedding_model_dict[model_name], model_kwargs=model_kwargs)

    return embedding


# 获取余弦相似度函数，torch.nn
def get_simfunc():
    simfunc = CosineSimilarity(dim=0, eps=1e-6)
    return simfunc
