import sys
from utils.general.preprocessing import *
from compute.compute_tfidf_affinity import compute_tfidf_affinity
from compute.compute_wmd_gensim_affinity import compute_wmd_gensim_affinity
from compute.compute_spacy_affinity import compute_spacy_affinity

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

	for algo in algorithms_to_process:
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


		function_to_call = "compute_" + algo['name'] + "_affinity"
		similarity = eval(function_to_call)(x_list_postprocess, y_list_postprocess, loaded_models[algo['name']],hyper_parameters,logger)*100
		#similarity = compute_tfidf_affinity(x_list_postprocess, y_list_postprocess, loaded_models[algo['name']])
		logger.info(similarity.shape)
		logger.info('Similarity Returned %s' %similarity)

		#print (x_list_postprocess)
		#print (y_list_postprocess)
		#compute_tfidf_affinity()

	return True
