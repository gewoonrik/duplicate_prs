import pickle

from DuplicatePRs.dataset import load_csv, get_tokenized_data
from DuplicatePRs import config
from collections import Counter

min_count = 5

tpr1s, tpr2s, _ = get_tokenized_data(load_csv(config.training_dataset_file))
valpr1s, valpr2s, _ = get_tokenized_data(load_csv(config.validation_dataset_file))

total = tpr1s+tpr2s+valpr1s+valpr2s

flat = [item for sublist in total for item in sublist]

vocab = Counter(flat)


filtered_vocab = {}
i = 0
for token, count in vocab.iteritems():
    if count >= min_count:
        filtered_vocab[token] = i
        i += 1

f = open(config._current_path+"/baseline/vocab.py", "w")
pickle.dump(filtered_vocab,f)
f.close()