import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def compute_weighted_tfidf_affinity(model_name,x,y,word_embeddings,hyper_parameters,logger,general_config):
    embedding_dim = hyper_parameters["dimensions"]
    tfid_vectorizer = create_tfidf_vector(x,y,hyper_parameters,logger,general_config)
    word_tf_idf_score_dict = dict(zip(tfid_vectorizer.get_feature_names(), tfid_vectorizer.idf_))
    x_embed = []
    zero_dim = np.zeros((embedding_dim,))

    for item in x:
        sum = np.zeros((embedding_dim,))
        for word in item:
            word_embed = word_embeddings.get(word,zero_dim)
            word_tf_idf_score = word_tf_idf_score_dict.get(word, 1)
            weighted_score = word_tf_idf_score * word_embed
            sum = sum + weighted_score
        x_embed.append(sum/len(item))

    y_embed = []

    for item in y:
        sum = np.zeros((embedding_dim,))
        for word in item:
            word_embed = word_embeddings.get(word,zero_dim)
            word_tf_idf_score = word_tf_idf_score_dict.get(word, 1)
            weighted_score = word_tf_idf_score * word_embed
            sum = sum + weighted_score
        y_embed.append(sum/len(item))

    similarity_matrix = cosine_similarity(x_embed, y_embed)
    return similarity_matrix

def create_tfidf_vector(x,y,hyper_parameters,logger,general_config):
    if 'ngram_range' in hyper_parameters.keys():
        ngram_range = hyper_parameters['ngram_range']
    else:
        ngram_range = (int(general_config['models_details']['tfidf']['parameters']['ngram_range_start']),
                       int(general_config['models_details']['tfidf']['parameters']['ngram_range_end']))
    if 'min_df' in hyper_parameters.keys():
        min_df = hyper_parameters['min_df']
    else:
        min_df = int(general_config['models_details']['tfidf']['parameters']['min_df'])

    tfid_vectorizer = TfidfVectorizer(decode_error='replace', encoding='utf-8', ngram_range=ngram_range, min_df=min_df)
    tfid_vectorizer.fit_transform(x)
    return tfid_vectorizer




