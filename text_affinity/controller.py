import os
import sys
import traceback
import pandas as pd
import yaml
from utils.logger.logger import get_logger
from utils.storage.check_path_exist import check_path_exist
from utils.general.common import *
import pandas
import json
from compute.computation_engine import computation_engine

input_data = {"source" :["Obama speaks to media in Illinois","President greets the press in Chicago"],
              "target":["President greets the press in Chicago"],
              "algorithms" :[{"name":"wmd_gensim","preprocess":True,"hyper_parameters":[]}], #{"name":"wmd_gensim", "preprocess":True}],
			  #"algorithms" :[{"name":"tfidf","preprocess":True,"hyper_parameters":[],"preprocess_steps":['convert_lower']},
			  #               #{"name":"wmd_gensim","preprocess":True,"hyper_parameters":[],"preprocess_steps":[]}],
			  #				{"name":"spacy","preprocess":True,"hyper_parameters":[],"preprocess_steps":[]}],
              "custom_stopwords" : ["tanveer"]}

'''
input_data = {"source" :["hello"],
              "target":["hello"],
              "algorithms" :[{"name":"spacy","preprocess":True,"hyper_parameters":[]}], #{"name":"wmd_gensim", "preprocess":True}],
			  #"algorithms" :[{"name":"tfidf","preprocess":True,"hyper_parameters":[],"preprocess_steps":['convert_lower']},
			  #               #{"name":"wmd_gensim","preprocess":True,"hyper_parameters":[],"preprocess_steps":[]}],
			  #				{"name":"spacy","preprocess":True,"hyper_parameters":[],"preprocess_steps":[]}],
              "custom_stopwords" : ["tanveer"]}
'''

def setup(general_config):

	path_info = check_path_exist(general_config['generals']['storage_path'])

	if path_info['status_code'] == 1:
		print("Aborting Startup Path Doesn't Exist")
		sys.exit(-1)

	storage_path = path_info['path']

	logger = get_logger(general_config['generals'])

	logger.info("deploying text affinity service")
	logger.info("storage path %s" %storage_path)

	path_info = check_path_exist(general_config['generals']['model_path'])
	if path_info['status_code'] == 1:
		logger.info("path did not exist for model files creating path %s" %path_info['path'])
		os.makedirs(path_info['path'])
	else:
		logger.info("Models are located at %s" %path_info['path'])

	path_info = check_path_exist(general_config['stopwords']['path'])
	if path_info['status_code'] == 0:
		stopwords_df = pd.read_csv(path_info['path'])
		stopwords_df = stopwords_df[stopwords_df['Consider'] != 'No']
		stopwords = stopwords_df['Stopwords'].to_list()
		logger.info("Total Number Of Stopwords loaded: {}".format(len(stopwords)))
	else:
		logger.error("Please initialize a file for stopwords @ %s" %path_info['path'])
		sys.exit(-1)
	return (storage_path,stopwords,logger)

def main():
	general_config = read_general_config()
	storage_path,stopwords,logger = setup(general_config)
	logger.info("Sample Stop Words Are {}".format(stopwords[1:10]))
	loaded_models = load_models(general_config,logger)

	x = computation_engine(input_data, loaded_models, stopwords, storage_path, general_config, logger)

if __name__ == "__main__":
	main()
