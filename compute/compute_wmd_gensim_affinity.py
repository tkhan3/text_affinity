import numpy
import math

def compute_wmd_gensim_affinity(model_name,x,y,model,hyper_parameters,logger,general_config):
	logger.info('executing %s based similarity' % model_name)
	distance_matrix = numpy.zeros((len(x), len(y)))
	index_x = 0
	index_y = 0
	for sent_x in x:
		words_x = sent_x.split()
		index_y = 0
		for sent_y in y:
			words_y = sent_y.split()
			distance_matrix[index_x,index_y] = model.wmdistance(words_x,words_y)
			if math.isinf(distance_matrix[index_x,index_y]):
				distance_matrix[index_x, index_y] = 9999
			print (distance_matrix[index_x,index_y])
			index_y = index_y + 1
		index_x = index_x + 1

	return distance_matrix
