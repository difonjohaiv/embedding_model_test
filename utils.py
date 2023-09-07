import torch
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from torch.nn import CosineSimilarity

EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"
model_kwargs = {'device': EMBEDDING_DEVICE}

embedding_model_dict = {
    "text2vec-base": "models/text2vec-base-chinese",
    "text2vec-large": "models/text2vec-large-chinese",
    "m3e-base": "models/m3e-base",
    "bge-large": "models/bge-large-zh"
}


# 获取嵌入模型
def get_embedding(model_name="text2vec-large"):
    embedding = HuggingFaceEmbeddings(
        model_name=embedding_model_dict[model_name], model_kwargs=model_kwargs)
    print("获取的嵌入模型是:", model_name)
    return embedding


# 获取余弦相似度函数，torch.nn
def get_simfunc():
    simfunc = CosineSimilarity(dim=0, eps=1e-6)
    return simfunc
