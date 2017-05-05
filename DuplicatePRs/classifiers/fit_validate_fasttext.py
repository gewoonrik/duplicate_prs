import argparse

from DuplicatePRs import config
from DuplicatePRs.classifiers.preprocessing import preprocess
from DuplicatePRs.dataset import load_csv, get_tokenized_data
import fasttext
from keras.models import load_model
import numpy as np
import keras.backend as K

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def acc(y_true, y_pred, cutoff):
    y_true = np.asarray([int(x) for x in y_true])
    y_res = []
    for y in y_pred:
        if y < cutoff:
            y_res.append(1)
        else:
            y_res.append(0)
    return np.mean(np.equal(y_true, np.asarray(y_res)))



fasttext_model = fasttext.load_model("fasttext/model.bin")

lines = load_csv(config.validation_dataset_file)
val_1, val_2, val_labels = get_tokenized_data(lines)

val_1 = preprocess(val_1, fasttext_model, config.maxlen, config.maxlen)
val_2 = preprocess(val_2, fasttext_model, config.maxlen, config.maxlen)


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='')

args = parser.parse_args()

model = load_model("classifier_models/cnn_fasttext_classifier-0.79271.hdf5",{"contrastive_loss":contrastive_loss})


results = model.predict([val_1, val_2], batch_size=50)

for i in range(1, 20, 1):
    i = i/10.0
    print("test "+str(i))
    print("result acc: "+str(acc(val_labels, results, i)))



