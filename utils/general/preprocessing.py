import re
from tqdm import tqdm
import traceback

def convert_lower(text, stopwords, spacy):
	return text.lower()


def remove_stopwords(text, stopwords, spacy):
	text = " ".join([i for i in text.split(' ') if i not in stopwords])
	return text


def remove_symbols(text, stopwords, spacy):
	text = re.sub(r'[^\x00-\x7F]+', ' ', text)
	text = re.sub(r'[^a-zA-Z ]', ' ', text)
	return text


def lemmatize(text, stopwords, spacy):
	spacy.max_length = len(text) + 100
	doc = spacy(text)
	lemma_text = " ".join([token.lemma_ for token in doc])
	lemma_text = lemma_text.replace('-PRON-', ' ')
	return lemma_text.strip()


def preprocessing(texts,stopwords,spacy,preprocessing_steps):
	for step in preprocessing_steps:
		texts = [eval(step)(text, stopwords, spacy) for text in tqdm(texts)]
	return texts