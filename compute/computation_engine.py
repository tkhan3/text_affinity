import sys
from utils.general.preprocessing import *
from compute.compute_tfidf_affinity import compute_tfidf_affinity
from compute.compute_wmd_gensim_affinity import compute_wmd_gensim_affinity
from compute.compute_spacy_affinity import compute_spacy_affinity
from compute.compute_use_affinity import compute_use_affinity
from compute.compute_avg_wordembedding_affinity import compute_avg_wordembedding_affinity
from compute.compute_weighted_tfidf_affinity import compute_weighted_tfidf_affinity

def computation_engine(input_data, loaded_models, stopwords, storage_path, general_config, logger):
	x_list = input_data.get('source')
	y_list = input_data.get('target')
	custom_stopwords = input_data.get("custom_stopwords")
	loaded_models_names = list(loaded_models.keys())
	algorithms_to_process = input_data.get('algorithms')

	logger.info('similarity request body %s' % input_data)

	algorithms_to_process_names = [algo['name'] for algo in algorithms_to_process]
	print (algorithms_to_process_names)
	# if the requested algorithm doesn't exist then return error
	if not set(algorithms_to_process_names).issubset(set(general_config['models_load']['load_all'])):
		logger.error("Requested Model is not supported in Text Affinity %s" % algorithms_to_process)
		sys.exit(-1)

	if not set(algorithms_to_process_names).issubset(set(loaded_models_names)):
		logger.error("Requested model was not loaded on startup, Add the model in config and restart the server")
		sys.exit(-1)

	if custom_stopwords:
		stopwords.extend(custom_stopwords)
		logger.info("Appended Requested Stop Words To Our Dictionary")

	similarity = []
	for algo in algorithms_to_process:
		similarity_map = dict()
		logger.info('Processing Algorithm %s' %algo['name'])
		try:
			if algo['preprocess']:
				if algo['preprocess_steps']:
					preprocess_steps = algo['preprocess_steps']
					logger.info('Using Preprocessing Steps Specified in API Call %s' %preprocess_steps)
				else:
					preprocess_steps = general_config['models_details'][algo['name']]['preprocessing_steps']
					logger.info('Using Preprocessing Steps Specified in Configuration File %s' %preprocess_steps)
				x_list_postprocess = preprocessing(x_list, stopwords, loaded_models['spacy'],preprocess_steps)
				y_list_postprocess = preprocessing(y_list, stopwords, loaded_models['spacy'], preprocess_steps)
			else:
				x_list_postprocess = x_list
				y_list_postprocess = y_list
		except KeyError:
			x_list_postprocess = x_list
			y_list_postprocess = y_list
			logger.info("Preprocessing Skipped As Not Defined At Model Level")

		hyper_parameters = {}
		try:
			if algo['hyper_parameters']:
				logger.info("using hyperparameters defined in API call")
				hyper_parameters = algo['hyper_parameters']
		except KeyError:
			logger.info("Using default hyperparameters from config file")

		loaded_model_obj = loaded_models[algo['name']]
		function_name = algo['name']

		if algo["name"] in general_config["models_details"]["word_embedding_models"]["name"]:
			function_name = general_config["models_details"]["word_embedding_models"]["compute_function_name"]
			hyper_parameters["dimensions"] = general_config["models_details"][algo["name"]]["parameters"]["dimension"]
		elif algo["name"] in general_config["models_details"]["dependent_model"].keys():
			try:
				prerequisite_model = general_config["models_details"]["dependent_model"][algo["name"]]
				loaded_model_obj = loaded_models[prerequisite_model]
				hyper_parameters["dimensions"] = general_config["models_details"][prerequisite_model]["parameters"]["dimension"]
			except KeyError:
				logger.info("Dependent Model Is Not Loaded %s" %algo['name'])
				sys.exit(-1)

		function_to_call = "compute_" + function_name + "_affinity"
		similarity_matrix = eval(function_to_call)(algo['name'],x_list_postprocess, y_list_postprocess, loaded_model_obj,hyper_parameters,logger,general_config)
		#print (type(similarity_matrix))
		similarity_map[algo['name']] = similarity_matrix.tolist()
		#similarity = compute_tfidf_affinity(x_list_postprocess, y_list_postprocess, loaded_models[algo['name']])
		logger.info(similarity_matrix.shape)
		logger.info('Similarity Returned %s' %similarity_matrix)
		similarity.append(similarity_map)

		#print (x_list_postprocess)
		#print (y_list_postprocess)
		#compute_tfidf_affinity()

	logger.info(similarity)

	return similarity
