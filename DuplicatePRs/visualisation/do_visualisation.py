from keras.models import load_model, model_from_json
from DuplicatePRs import config
from DuplicatePRs.classifiers.preprocessing import get_preprocessed_generator
from DuplicatePRs.visualisation.visualize import visualize
import keras.backend as K
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


best_model = "best.h5"
f = open(config._current_path+"/classifier_models/cnn_euclidian/model.json")
json = f.read()
f.close()

model = model_from_json(json, {"contrastive_loss":contrastive_loss, "acc":acc})
model.load_weights(config._current_path+"/classifier_models/cnn_euclidian/"+best_model)
w2vec =  Word2Vec.load(config.doc2vec_model_directory+"doc2vec_word2vec_dbow_epoch9.model")
embeddings_model = w2vec.wv
# save memory
del w2vec

gen, steps, labels = get_preprocessed_generator(config.test_dataset_file, embeddings_model, config.embeddings_size, None, 1)

prs, label = gen.next()

print(visualize(model, prs[0]))

