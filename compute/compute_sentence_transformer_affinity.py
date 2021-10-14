from sklearn.metrics.pairwise import cosine_similarity

def compute_sentence_transformer_affinity(model_name,x,y,sentence_transformer,hyper_parameters,logger,general_config):
    x_embed = sentence_transformer.encode(x, show_progress_bar=True)
    y_embed = sentence_transformer.encode(y, show_progress_bar=True)
    similarity_matrix = cosine_similarity(x_embed, y_embed)
    return similarity_matrix