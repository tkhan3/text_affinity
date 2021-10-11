import numpy
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

def compute_use_affinity(x,y,use_model,hyper_parameters,logger):
	try:
		embedding_x = use_model(x)
		embedding_y = use_model(y)
		similarity_matrix = cosine_similarity(embedding_x, embedding_y)
		return similarity_matrix
	except Exception as e:
		logger.info("Error in USE Computation %s" %e)
		return None