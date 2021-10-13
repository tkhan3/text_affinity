import yaml
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.storage.check_path_exist import *
from gensim.models import Word2Vec,KeyedVectors
from gensim.test.utils import datapath
import spacy
import tensorflow_hub as hub
import numpy as np

def read_general_config(filename):
    filepath = "configuration/" + filename
    with open(filepath, "r") as general_config:
        return yaml.load(general_config, Loader=yaml.FullLoader)


## add an environment variable with LOAD_SELECTED_MODELS with comma separated list of models to be loaded.
def load_models(general_config, logger):
    models_to_load = list()
    if "LOAD_SELECTED_MODELS" in os.environ:
        models_to_load = os.environ['LOAD_SELECTED_MODELS'].split(",")
        logger.info("loading selected models ==> %s" % os.environ['LOAD_SELECTED_MODELS'])
    else:
        models_to_load = general_config["models_load"]["load_all"]
        logger.info("all models to get loaded ==> %s" % models_to_load)

    #models_to_load = ['tfidf','wmd_gensim']
    #models_to_load = ['tfidf']
    #models_to_load = ['wmd_gensim']
    models_to_load.append('spacy')
    loaded_models = {}

    for item in models_to_load:
        print(general_config["models_details"][item]["loader"])
        item_model = globals()[general_config["models_details"][item]["loader"]](general_config, logger,item)
        loaded_models[item] = item_model

    return loaded_models


def load_wmd_gensim(general_config, logger,model_name):
    path_info = check_path_exist(general_config['models_details']['wmd_gensim']['path'])
    if path_info['status_code'] == 1:
        logger.info('word2vec model doesn\'t exist at %s' %path_info['path'])

    model = KeyedVectors.load_word2vec_format(datapath(general_config['models_details']['wmd_gensim']['path']),binary=True)
    ##distance = model.wmdistance('Obama speaks to the media in Illinois','The president greets the press in Chicago')
    ##print (distance)
    ##model.fill_norms()
    ##model.most_similar('twitter')
    print (type(model))
    return model


def load_tfidf(general_config, logger,model_name):
    ngram_range = (int(general_config['models_details']['tfidf']['parameters']['ngram_range_start']),
                   int(general_config['models_details']['tfidf']['parameters']['ngram_range_end']))
    min_df = int(general_config['models_details']['tfidf']['parameters']['min_df'])
    tfid_vectorizer = TfidfVectorizer(decode_error='replace', encoding='utf-8', ngram_range=ngram_range, min_df=min_df)
    logger.info("tfidf initiated with ngram_range: {} and min_df: {}".format(ngram_range, min_df))
    return tfid_vectorizer


def load_spacy(general_config,logger,model_name):
    spacy_model = general_config["models_details"]["spacy"]["parameters"]["type"]
    try:
        wmd_spacy = spacy.load(spacy_model)
    except OSError as exp:
        print (exp)
        logger.info('Downloading the spacy model')
        spacy.cli.download(spacy_model)
        wmd_spacy = spacy.load(spacy_model)

    logger.info("Spacy Models Loaded")
    return wmd_spacy

def load_use(general_config,logger,model_name):

    use_model_path = general_config['models_details']['use']['path']
    path_info = check_path_exist(use_model_path)

    if path_info['status_code'] == 1:
        logger.info('USE model doesn\'t exist at %s' % path_info['path'])

    use_model = hub.load(use_model_path)
    logger.info("USE Model Loaded")
    return use_model

def load_word_embedding(general_config,logger,model_name):
    embedding_model_path = general_config['models_details'][model_name]['path']
    print (embedding_model_path)
    path_info = check_path_exist(embedding_model_path)

    if path_info['status_code'] == 1:
        logger.info('%s Model Doesnt exist at %s' %(model_name,embedding_model_path))

    embeddings = open(path_info["path"], encoding='utf-8')

    word_embedding = {}

    for line in embeddings:
        values = line.split()
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        word_embedding[word] = embedding

    embeddings.close()
    logger.info("Embedding Loaded For The Model %s" %model_name)

    return word_embedding
