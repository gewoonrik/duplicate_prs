import argparse
from functools import partial

from keras.models import load_model
import keras.backend as K

from DuplicatePRs import config
from DuplicatePRs.classifiers.preprocessing import preprocess, get_preprocessed_generator
from DuplicatePRs.dataset import load_csv, get_tokenized_data


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    # duplicates should be low, non duplicates should be high
    # so duplicates = 0, non duplicate = 1
    y_true = -1 * y_true + 1
    return K.mean((1 - y_true) * K.square(y_pred) +  y_true * K.square(K.maximum(margin - y_pred, 0)))


def acc(cutoff, y_true, y_pred):
    k_cutoff = K.ones_like(y_pred) * (0.5-cutoff)
    return K.mean(K.equal(y_true, K.round(K.clip(y_pred + k_cutoff,0,1))), axis=-1)


parser = argparse.ArgumentParser()
parser.add_argument('--embeddings_model', default='word2vec')
parser.add_argument('--model', default='')
parser.add_argument('--cutoff', default=0.5, type=float)


args = parser.parse_args()


if(args.embeddings_model == "word2vec"):
    from gensim.models import Word2Vec
    w2vec =  Word2Vec.load(config.doc2vec_model_directory+"doc2vec_word2vec_dbow_epoch9.model")
    embeddings_model = w2vec.wv
    # save memory
    del w2vec
else:
    import fasttext
    embeddings_model = fasttext.load_model("fasttext/model.bin")


lines = load_csv(config.test_dataset_file)
test_gen, test_steps, test_labels = get_preprocessed_generator(config.validation_dataset_file, embeddings_model, config.embeddings_size, None, 5)

acc_cutoff = partial(acc, args.cutoff)
model = load_model(config._current_path+"/classifier_models/"+args.model,{"contrastive_loss":contrastive_loss, "acc":acc_cutoff})


results = model.evaluate_generator(test_gen, test_steps)

print('Test score: ', results)

