import numpy as np
import math
from DuplicatePRs.dataset import get_tokenized_data, load_csv


cached_vectors = {}


def preprocess_text(text, embeddings_model, maxlen, embeddings_size):
    seq = text
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

def generator(prs1, prs2, labels, embeddings_model, embeddings_size, maxlen, batch_size):
    while True:
        for i in xrange(0, len(prs1), batch_size):
            prs1_sliced = prs1[i:i + batch_size]
            prs2_sliced = prs2[i:i + batch_size]
            labels_sliced = labels[i:i + batch_size]

            prs1_res = preprocess(prs1_sliced, embeddings_model, embeddings_size, maxlen)
            prs2_res = preprocess(prs2_sliced, embeddings_model, embeddings_size, maxlen)
            yield ([prs1_res, prs2_res], labels_sliced)


def get_preprocessed_generator(file, embeddings_model, embeddings_size, maxlen, batch_size):
    prs_1, prs_2, y = get_tokenized_data(load_csv(file))
    return generator(prs_1, prs_2, y, embeddings_model, embeddings_size, maxlen, batch_size), math.ceil(len(y)/batch_size)

