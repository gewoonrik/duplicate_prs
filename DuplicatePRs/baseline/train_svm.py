from __future__ import print_function

from multiprocessing import Pool

from scipy.sparse import lil_matrix
import itertools
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

from DuplicatePRs import config
from DuplicatePRs.dataset import get_tokenized_data_generator, load_csv
from gensim.corpora import Dictionary

tr_lines = load_csv(config.training_dataset_file)
val_lines = load_csv(config.validation_dataset_file)
tr_gen = get_tokenized_data_generator(tr_lines)
val_gen = get_tokenized_data_generator(val_lines)

print("loading dict")
dict = Dictionary().load(config._current_path+"/baseline/dict3")

nr_words = len(dict.token2id)

def line_to_bow(line):
    pr1,pr2,label = line
    return dict.doc2bow(pr1), dict.doc2bow(pr2), label
def dataset_to_bow(generator, length):
    matrix = lil_matrix((length, nr_words*2), dtype=int)
    labels = []
    p = Pool(10)
    bow = p.map(line_to_bow, generator)
    for i, (pr1,pr2,label) in enumerate(bow):
        print("creating matrix %s / %s" % (i, length), end='\r')

        pr1 = map(lambda x: x.decode('utf-8', 'ignore'), pr1)
        pr2 = map(lambda x: x.decode('utf-8', 'ignore'), pr2)
        pr1_bow = dict.doc2bow(pr1)
        pr2_bow = dict.doc2bow(pr2)
        for (id, count) in pr1_bow:
            matrix[i,id] = count
        for (id, count) in pr2_bow:
            matrix[i,id+nr_words] = count
        labels.append(label)
    return matrix, labels
print("creating matrix")
training_matrix, tr_labels = dataset_to_bow(tr_gen, len(tr_lines))

svm = LinearSVC(verbose=1)
print("fitting ")
svm.fit(training_matrix, tr_labels)

joblib.dump(svm, config._current_path+"/baseline/svm")

validation_matrix, val_labels = dataset_to_bow(val_gen, len(val_lines))
print("testing")
acc = svm.score(validation_matrix, val_labels)
print("accuracy " +str(acc))