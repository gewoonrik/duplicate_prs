import argparse

from keras.models import Model
from keras.layers import Lambda
from keras.layers import Input
from keras import backend as K
from keras.optimizers import Adam
from preprocessing import get_preprocessed_generator
from cnn_shared_model import conv_model
from DuplicatePRs import config


# Convolution
filters = 150


# Training
batch_size = 100
epochs = 150

def acc(y_true, y_pred):
    ones = K.ones_like(y_pred)
    return K.mean(K.equal(y_true, ones - K.clip(K.round(y_pred), 0, 1)), axis=-1)


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

parser = argparse.ArgumentParser()
parser.add_argument('--embeddings_model', default='word2vec')
parser.add_argument('--learning_rate', type=float, default=0.003)

args = parser.parse_args()


if(args.embeddings_model == "word2vec"):
    from gensim.models import Word2Vec
    embeddings_model =  Word2Vec.load(config.doc2vec_model_directory+"doc2vec_dbow_epoch9_notest.model")
    embeddings_model = embeddings_model.wv
else:
    import fasttext
    embeddings_model = fasttext.load_model(config.fasttext_model_directory+"fasttext/model.bin")


tr_gen, tr_steps = get_preprocessed_generator(config.training_dataset_file, embeddings_model, config.embeddings_size, config.maxlen, batch_size)
val_gen, val_steps = get_preprocessed_generator(config.validation_dataset_file, embeddings_model, config.embeddings_size, config.maxlen, batch_size)
te_gen, te_steps = get_preprocessed_generator(config.test_dataset_file, embeddings_model, config.embeddings_size, config.maxlen, batch_size)


print('Build model...')

pr_1 =  Input(shape=(config.maxlen, config.embeddings_size), dtype='float32')
pr_2 =  Input(shape=(config.maxlen, config.embeddings_size), dtype='float32')

out_1 = conv_model(pr_1)
out_2 = conv_model(pr_2)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([out_1, out_2])

model = Model(input=[pr_1, pr_2], output=distance)

optimizer = Adam(lr=args.learning_rate)

model.compile(loss=contrastive_loss,
              optimizer=optimizer, metrics=[acc])

print('Train...')


model.fit_generator(tr_gen, steps_per_epoch=tr_steps,
                    epochs=epochs,
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    workers=1)
score, acc = model.evaluate_generator(te_gen, steps=te_steps)
print('Test score:', score)
print('Test accuracy:', acc)

