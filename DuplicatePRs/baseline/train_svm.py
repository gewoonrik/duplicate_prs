from __future__ import print_function

from multiprocessing import Pool

from scipy.sparse import lil_matrix
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

from DuplicatePRs import config
from DuplicatePRs.dataset import load_csv, read_pickled, line_to_tokenized_files
from gensim.corpora import Dictionary

tr_lines = load_csv(config.training_dataset_file)
val_lines = load_csv(config.validation_dataset_file)
tr_gen = map(line_to_tokenized_files,tr_lines)
val_gen = map(line_to_tokenized_files,val_lines)


print("loading dict")
dict = Dictionary().load(config._current_path+"/baseline/dict_hard")

nr_words = len(dict.token2id)

def line_to_bow(line):
    pr1,pr2,label = line
    pr1 = read_pickled(pr1)
    pr2 = read_pickled(pr2)
    pr1 = map(lambda x: x.decode('utf-8', 'ignore'), pr1)
    pr2 = map(lambda x: x.decode('utf-8', 'ignore'), pr2)
    return dict.doc2bow(pr1), dict.doc2bow(pr2), label


def dataset_to_bow(generator, length):
    matrix = lil_matrix((length*2, nr_words*2), dtype=int)
    labels = []
    p = Pool(14)
    print("mapping")
    bow = p.map(line_to_bow, generator)
    print("done mapping")
    for i, (pr1_bow,pr2_bow,label) in enumerate(bow):
        print("creating matrix %s / %s" % (i, length), end='\r')

        for (id, count) in pr1_bow:
            matrix[2*i,id] = count
            matrix[2*i+1,id+nr_words] = count
        for (id, count) in pr2_bow:
            matrix[2*i,id+nr_words] = count
            matrix[2*i+1,id] = count
        labels.append(label)
        labels.append(label)
    return matrix, labels
print("creating matrix")
training_matrix, tr_labels = dataset_to_bow(tr_gen, len(tr_lines))

scaler = preprocessing.StandardScaler(with_mean=False).fit(training_matrix)

training_matrix = scaler.transform(training_matrix)


params = [1,0.1,0.01,0.001,0.0001,0.0001,0.00001,0.000001, 0.0000001]

results = []
for c in params:
    svm = LinearSVC(verbose=1, max_iter=10000, C=c)
    print("fitting "+str(c))
    svm.fit(training_matrix, tr_labels)

    joblib.dump(svm, config._current_path+"/baseline/svm_hard-c-"+str(c))

    validation_matrix, val_labels = dataset_to_bow(val_gen, len(val_lines))
    print("testing")
    acc = svm.score(scaler.transform(validation_matrix), val_labels)
    print("accuracy " +str(acc))
    results.append(acc)
print(params)
print(results)