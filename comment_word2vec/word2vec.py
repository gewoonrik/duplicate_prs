import os
import logging
from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Sentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.lower().split()

sentences = Sentences('comments_preprocessed')

min_count = 5
size = 150
window = 4
model = Word2Vec(sentences, min_count=min_count, size=size, window=window, workers = 16)

model.save('word2vec.model')

print(model.most_similar(positive=["duplicate"]))
