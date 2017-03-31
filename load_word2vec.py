#!/usr/bin/python
from gensim.models import Word2Vec
import sys
model =  Word2Vec.load("word2vec.model")
print(model.most_similar(positive=[sys.argv[1]], topn=20))
