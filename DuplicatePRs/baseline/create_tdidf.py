import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

from DuplicatePRs.dataset import load_csv, get_tokenized_data
from DuplicatePRs import config
from collections import Counter

min_count = 5

tpr1s, tpr2s, _ = get_tokenized_data(load_csv(config.training_dataset_file))
valpr1s, valpr2s, _ = get_tokenized_data(load_csv(config.validation_dataset_file))

total = tpr1s+tpr2s+valpr1s+valpr2s

tfidf = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x,
                        analyzer = "word",
                        lowercase = False,
                        min_df = 5)
tfidf = tfidf.fit(total)

f = open(config._current_path+"/baseline/tfidf.model", "w")
pickle.dump(tfidf, f)
f.close()