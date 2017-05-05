from gensim.models.wrappers import FastText
from DuplicatePRs import config

model = FastText.train("fasttext/fastText/fasttext", "fasttext/training_data.txt", "fasttext/model","skipgram", config.embeddings_size, threads=15,iter = 1)
