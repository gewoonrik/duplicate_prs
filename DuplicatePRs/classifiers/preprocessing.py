import threading
from multiprocessing import Pool
from functools import partial
import numpy as np
import math
from DuplicatePRs.dataset import get_tokenized_data, load_csv, line_to_tokenized_files, read_pickled

cached_vectors = {}


def preprocess_text(seq, embeddings_model, maxlen, embeddings_size):
    seq = seq[0:maxlen-1]
    res = np.zeros((maxlen, embeddings_size))
    seq_length = len(seq)
    offset = maxlen - seq_length

    i = 0
    for elem in seq:
        try:
            if elem in cached_vectors:
                res[i+offset] = cached_vectors[elem]
            else:
                vec = embeddings_model[elem]
                res[i+offset] = vec
                cached_vectors[elem] = vec
        except KeyError:
            #just keep zeros if we don't have a vector
            pass
        i +=1

    return res


def preprocess(texts, embeddings_model, embeddings_size, maxlen):
    results = np.zeros((len(texts),maxlen, embeddings_size))
    for i, text in enumerate(texts):
        results[i] = preprocess_text(text, embeddings_model, maxlen, embeddings_size)
    return results

def read_and_preprocess(embeddings_model, embeddings_size, maxlen, pr_file):
    content = read_pickled(pr_file)
    return preprocess_text(content, embeddings_model, maxlen, embeddings_size)

class DataIterator:
    def __init__(self, prs1, prs2, labels, embeddings_model, embeddings_size, maxlen, batch_size):
        self.prs1 = prs1
        self.prs2 = prs2
        self.labels = labels
        self.embeddings_model = embeddings_model
        self.embeddings_size = embeddings_size
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.lock = threading.Lock()
        self.i = 0
        self.steps = math.ceil(len(labels)/batch_size)
        self.p = Pool(8)
        self.preprocess = partial(read_and_preprocess, embeddings_model, embeddings_size, maxlen)

    def __iter__(self):
        return self

    def reset(self):
        self.i = 0

    def next(self):
        with self.lock:
            i = self.i
            self.i += 1
            if self.i > self.steps:
                self.reset()

        cur = i * self.batch_size
        prs1_sliced = self.prs1[cur:cur + self.batch_size]
        prs2_sliced = self.prs2[cur:cur + self.batch_size]
        labels_sliced = self.labels[cur:cur + self.batch_size]

        prs1_sliced = self.p.map(self.preprocess, prs1_sliced)
        prs2_sliced = self.p.map(self.preprocess, prs2_sliced)

        return ([np.asarray(prs1_sliced), np.asarray(prs2_sliced)], labels_sliced)


def lines_to_tokenized_files(lines):
    prs1 = []
    prs2 = []
    labels = []
    for line in lines:
        pr1, pr2, label = line_to_tokenized_files(line)
        prs1.append(pr1)
        prs2.append(pr2)
        labels.append(label)
    return prs1, prs2, labels


def get_preprocessed_generator(file, embeddings_model, embeddings_size, maxlen, batch_size):
    prs_1, prs_2, y = lines_to_tokenized_files(load_csv(file))
    return DataIterator(prs_1, prs_2, y, embeddings_model, embeddings_size, maxlen, batch_size), math.ceil(len(y)/batch_size)

