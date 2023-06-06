import nltk
import numpy as np
import sys
from joblib import load
from gensim.models import Word2Vec
from sklearn.cluster import KMeans

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize_text(text):
  return set([lemmatizer.lemmatize(w, 'v') for w in w_tokenizer.tokenize(text)])


def vectorize_single(text, model):
      zero_vector = np.zeros(model.vector_size)
      vectors = []
      for token in text:
          if token in model.wv:
              try:
                  vectors.append(model.wv[token])
              except KeyError:
                  continue
      if vectors:
          vectors = np.asarray(vectors)
          avg_vec = vectors.mean(axis=0)
          return avg_vec
      else:
          return zero_vector

def load_reqs():
  kmeans_model = load("kmeans.model")
  vect_model = load("vect_model.model")
  return (kmeans_model, vect_model)
def predict(text, model, vect_model):
  text = text.lower()
  text = text.replace(r'[^A-Za-z]+', ' ')
  text = text.replace('withdrawal', 'withdraw')
  text = text.replace('withdraw against inter branch cash deposit', 'withdraw against inter branch cash')
  text = text.replace('withdraw against cash deposit', 'withdraw against cash')
  text = text.replace('withdraw against deposit', 'withdraw against')
  lem_tok = lemmatize_text(text)
  vect = vectorize_single(lem_tok, vect_model)
  return model.predict([vect])


if __name__ == "__main__":
  (model, vect_model) = load_reqs()
  print('cluster ', predict(sys.argv[1], model, vect_model))