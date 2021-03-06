from __future__ import print_function

import argparse
import pickle
from tqdm import tqdm

from keras.models import load_model
import keras.backend as K
from DuplicatePRs import config
from DuplicatePRs.classifiers.preprocessing import preprocess
from DuplicatePRs.dataset import load_csv, get_tokenized_files, read_pickled
from gensim.models import Word2Vec



def acc(y_true, y_pred):
    ones = K.ones_like(y_pred)
    return K.mean(K.equal(y_true, ones - K.clip(K.round(y_pred), 0, 1)), axis=-1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    # duplicates should be low, non duplicates should be high
    # so duplicates = 0, non duplicate = 1
    y_true = -1 * y_true + 1
    return K.mean((1 - y_true) * K.square(y_pred) +  y_true * K.square(K.maximum(margin - y_pred, 0)))
parser = argparse.ArgumentParser()
parser.add_argument('--embeddings_model', default='word2vec')

args = parser.parse_args()

def save(file, result):
    if args.embeddings_model == "word2vec":
        processed_pr_file = file.replace("_tokenized","_word2vec2doc_preprocessed")
    else:
        processed_pr_file = file.replace("_tokenized","_fasttext2doc_preprocessed")

    with open(processed_pr_file, 'w') as f:
        pickle.dump(result, f)



model_path = ""
if args.embeddings_model != "word2vec":
    model_path = "_fasttext"
else:
    model_path = "_word2vec"
model = load_model(config._current_path+"/classifier_models/cnn_euclidian"+model_path+"_hard/best.hdf5", {"contrastive_loss":contrastive_loss, "acc":acc})

# take only the shared CNN model :)
model = model.layers[-2]

train = load_csv(config.training_dataset_file)
val = load_csv(config.validation_dataset_file)
test = load_csv(config.test_dataset_file)

tr_files = get_tokenized_files(train)
val_files = get_tokenized_files(val)
te_files  = get_tokenized_files(test)

total = tr_files+val_files+te_files

if(args.embeddings_model == "word2vec"):
    from gensim.models import Word2Vec
    w2vec =  Word2Vec.load(config.doc2vec_model_directory+"doc2vec_word2vec_dbow_hard_epoch9.model")
    embeddings_model = w2vec.wv
    # save memory
    del w2vec
else:
    print("doing fasttext")
    import fasttext
    embeddings_model = fasttext.load_model(config.fasttext_model_directory+"/model.bin")

batch_size = 50
for i in tqdm(range(0, len(total), batch_size)):
#    print("doing nr "+str(i)+" of "+str(len(total)))
    files = total[i:i+batch_size]
    tokenized = map(read_pickled, files)
    lengths = map(len, tokenized)
    maxlen = min(max(lengths),config.maxlen)
    w2vec_files = preprocess(tokenized, embeddings_model, config.embeddings_size, maxlen)
    results = model.predict(w2vec_files)
    for i in range(len(files)):
        save(files[i], results[i])


