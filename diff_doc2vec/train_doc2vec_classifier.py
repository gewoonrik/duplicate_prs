import pickle

import numpy as np
from gensim.models.doc2vec import TaggedDocument
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.layers import Input, merge, Dense, Dropout
from keras.models import Model

from diff_scripts.load_data import load_data, lines_to_files
from tokenize import tokenize,filter_diff_lines


class Documents(object):
    def __init__(self, files):
        self.files = files

    def __iter__(self):
        for file in self.files:
            f = open(file, "r")
            content = f.read()
            f.close()
            yield TaggedDocument(words = tokenize(filter_diff_lines(content)), tags = [file])



def read(file):
    f = open(file, "r")
    content = pickle.load(f)
    f.close()
    return content

def load_docs2vec(files):
    prs1 = []
    prs2 = []
    labels = []
    for (pr1, pr2, is_dup) in files:
        vec_1 = read(pr1)
        vec_2 = read(pr2)
        prs1.append(vec_1)
        prs2.append(vec_2)
        labels.append(is_dup)
    return prs1, prs2, labels

print("loading files")
training = lines_to_files(load_data("training_with_negative_samples.csv"))
validation = lines_to_files(load_data("validation_with_negative_samples.csv"))
test = lines_to_files(load_data("test_with_negative_samples.csv"))


print("loading data")
tr_1, tr_2, tr_labels = load_docs2vec(training)
tr_1 = np.asarray(tr_1)
tr_2 = np.asarray(tr_2)
tr_labels = np.asarray(tr_labels)

val_1, val_2, val_labels = load_docs2vec(validation)
val_1 = np.asarray(val_1)
val_2 = np.asarray(val_2)
val_labels = np.asarray(val_labels)

test_1, test_2, test_labels = load_docs2vec(test)
test_1 = np.asarray(test_1)
test_2 = np.asarray(test_2)
test_labels = np.asarray(test_labels)


pr1 = Input(shape=(300,), dtype='float32', name='pr1_input')
pr2 = Input(shape=(300,), dtype='float32', name='pr2_input')
x = merged = merge([pr1, pr2], mode='concat')
x = Dropout(0.4)(x)
x = Dense(600, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(600, activation='relu')(x)
x = Dropout(0.4)(x)
main_output = Dense(1, activation='sigmoid', name='output')(x)

model = Model(input=[pr1, pr2], output=[main_output])


model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy', 'fmeasure'])


csv_logger = CSVLogger('trainingslog_doc2vec.csv')
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

print("train")
model.fit([tr_1, tr_2], tr_labels, batch_size=100, nb_epoch=20,
          validation_data=([val_1, val_2], val_labels), callbacks=[csv_logger,early_stopping])

results = model.evaluate([test_1, test_2], test_labels, batch_size=100)
print('Test results: ', results)
print('On metrics: ', model.metrics_names)
