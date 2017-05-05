#!/usr/bin/python
from gensim.models import Word2Vec
import sys
model =  Word2Vec.load("word2vec.model")
topn = 20
if sys.argv[2] != None:
	topn = int(sys.argv[2])
print(model.most_similar(positive=[sys.argv[1]], topn=topn))
