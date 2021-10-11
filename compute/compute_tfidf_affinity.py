from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
print (" i am here ")
def compute_tfidf_affinity(x,y,tfid_vectorizer,hyper_parameters,logger):

    if len(hyper_parameters) > 0:
        tfid_vectorizer = TfidfVectorizer(decode_error='replace', encoding='utf-8', ngram_range=hyper_parameters['ngram_range'],
                                          min_df=hyper_parameters['min_df'])
    logger.info('executing tfidf similarity')
    sparse_matrix = tfid_vectorizer.fit_transform(x)
    sparse_matrix2 = tfid_vectorizer.transform(y)
    similarity_matrix = cosine_similarity(sparse_matrix, sparse_matrix2)
    return similarity_matrix

def main():
    print ("i am here")

if __name__ == "__main__":
    main()