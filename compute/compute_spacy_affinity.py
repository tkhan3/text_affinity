import numpy
from tqdm import tqdm

def compute_spacy_affinity(x,y,spacy,hyper_parameters,logger):
    logger.info('executing spacy based similarity')
    try:
        similarity_matrix = numpy.zeros((len(x),len(y)))
        x_obj = [spacy(text) for text in x]
        y_obj = [spacy(text) for text in y]
        y_index = 0
        for each_y in tqdm(y_obj):
            x_index =0
            for each_x in x_obj:
                try:
                    score = each_y.similarity(each_x)
                except:
                    score = '9999'
                similarity_matrix[x_index][y_index] = score
                x_index = x_index + 1
            y_index = y_index + 1
        logger.info(similarity_matrix)
        return similarity_matrix
    except Exception as e:
        raise e