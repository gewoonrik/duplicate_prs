import argparse
from keras.models import load_model
import keras.backend as K

from DuplicatePRs import config
from DuplicatePRs.classifiers.preprocessing import preprocess
from DuplicatePRs.dataset import load_csv, get_tokenized_data


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def acc(y_true, y_pred):
    ones = K.ones_like(y_pred)
    return K.mean(K.equal(y_true, ones - K.clip(K.round(y_pred), 0, 1)), axis=-1)


parser = argparse.ArgumentParser()
parser.add_argument('--embeddings_model', default='word2vec')
parser.add_argument('--learning_rate', type=float, default=0.003)
parser.add_argument('--model', default='')


args = parser.parse_args()


if(args.embeddings_model == "word2vec"):
    from gensim.models import Word2Vec
    embeddings_model =  Word2Vec.load("doc2vec_models/doc2vec_dbow_epoch9_notest.model")
else:
    import fasttext
    embeddings_model = fasttext.load_model("fasttext/model.bin")


lines = load_csv("validation_with_negative_samples.csv")
test_1, test_2, test_labels = get_tokenized_data(lines)

test_1 = preprocess(embeddings_model, test_1, config.embeddings_size, config.maxlen)
test_2 = preprocess(embeddings_model, test_2, config.embeddings_size, config.maxlen)


model = load_model("classifier_models/cnn_fasttext_classifier-0.79271.hdf5",{"contrastive_loss":contrastive_loss, "acc":acc})


results = model.evaluate([test_1, test_2], test_labels, batch_size=100)

print('Test score: ', results)

