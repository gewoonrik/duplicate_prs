import threading

import keras
import numpy as np
import math
from DuplicatePRs.dataset import get_tokenized_data, load_csv, line_to_tokenized_files, read_pickled



def preprocess_text(text, embeddings_model, maxlen, embeddings_size):
    seq = text
    seq = seq[0:maxlen]
    res = np.zeros((maxlen, embeddings_size))
    seq_length = len(seq)
    offset = maxlen - seq_length

    i = 0
    for elem in seq:
        try:
            res[i+offset] = embeddings_model[elem]
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


class DataIterator:
    def __init__(self, prs1, prs2, labels, embeddings_model, embeddings_size, maxlen, batch_size, categorical_labels= False):
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
        self.categorical_labels = categorical_labels

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

        if self.maxlen == None:
            maxlen = max(map(len,prs1_sliced+prs2_sliced))
        else:
            maxlen = self.maxlen
        prs1_res = preprocess(prs1_sliced, self.embeddings_model, self.embeddings_size, maxlen)
        prs2_res = preprocess(prs2_sliced, self.embeddings_model, self.embeddings_size, maxlen)
        labels = np.concatenate([labels_sliced,labels_sliced])
        if self.categorical_labels:
            labels = keras.utils.to_categorical(labels,2)
        return ([np.concatenate([prs1_res,prs2_res]), np.concatenate([prs2_res, prs1_res])], labels)


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


def get_preprocessed_generator(file, embeddings_model, embeddings_size, maxlen, batch_size, cutoff=False, categorical_labels=False):
    print("loading data into memory")
    prs_1, prs_2, y = get_tokenized_data(load_csv(file), maxlen, cutoff)
    print("starting iterator")
    return DataIterator(prs_1, prs_2, y, embeddings_model, embeddings_size, maxlen, batch_size, categorical_labels), math.ceil(len(y)/(batch_size*1.0)), y

