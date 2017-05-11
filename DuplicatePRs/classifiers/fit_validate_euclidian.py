import argparse

from DuplicatePRs import config
from DuplicatePRs.classifiers.preprocessing import preprocess, get_preprocessed_generator
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


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='')
parser.add_argument('--embeddings_model', default='word2vec')


args = parser.parse_args()

if(args.embeddings_model == "word2vec"):
    from gensim.models import Word2Vec
    w2vec =  Word2Vec.load(config.doc2vec_model_directory+"doc2vec_word2vec_dbow_epoch9.model")
    embeddings_model = w2vec.wv
    # save memory
    del w2vec
else:
    import fasttext
    embeddings_model = fasttext.load_model(config.fasttext_model_directory+"fasttext/model.bin")
def get_label(line):
    owner, repo, pr1, pr2, is_dup = line.split(",")
    return is_dup

lines = load_csv(config.validation_dataset_file)

val_labels = map(get_label, lines)
val_gen, val_steps = get_preprocessed_generator(config.validation_dataset_file, embeddings_model, config.embeddings_size, config.maxlen, 50)



model = load_model(config._current_path+"/classifier_models/"+args.model,{"contrastive_loss":contrastive_loss})


results = model.predict_generator(val_gen, val_steps, verbose=1)

for i in range(1, 20, 1):
    i = i/10.0
    print("test "+str(i))
    print("result acc: "+str(acc(val_labels, results, i)))



