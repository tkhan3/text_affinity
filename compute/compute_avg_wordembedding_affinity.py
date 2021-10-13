import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_avgsent_embeddings(x,y,word_embeddings,embedding_dim):
    x_embed = []
    for i in x:
        if len(i.strip())!=0:
            v = sum([word_embeddings.get(w, np.zeros((embedding_dim,))) for w in i.split()]) / (len(i.split()) + 0.001)
        else:
            v = np.zeros((embedding_dim,))
        x_embed.append(v)
    y_embed = []
    for i in y:
        if len(i.strip())!=0:
            v = sum([word_embeddings.get(w, np.zeros((embedding_dim,))) for w in i.split()]) / (len(i.split()) + 0.001)
        else:
            v = np.zeros((embedding_dim,))
        y_embed.append(v)
    return x_embed,y_embed

def compute_avg_wordembedding_affinity(model_name,x,y,word_embeddings,hyper_parameters,logger):
    embedding_dim = hyper_parameters["dimensions"]
    try:
        x_embed,y_embed = compute_avgsent_embeddings(x,y,word_embeddings,embedding_dim)
        similarity_matrix = cosine_similarity(x_embed, y_embed)
        return similarity_matrix
    except Exception as e:
        raise e