import argparse

from keras import Input
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Conv1D, GlobalMaxPooling1D, Activation, dot, Lambda, MaxPooling1D
import keras.backend as K
from keras.optimizers import Adam

from DuplicatePRs import config
from DuplicatePRs.classifiers.attention_layer import AttentionLayer
from DuplicatePRs.classifiers.preprocessing import get_preprocessed_generator

batch_size = 10
epochs = 150


parser = argparse.ArgumentParser()
parser.add_argument('--embeddings_model', default='word2vec')
parser.add_argument('--learning_rate', type=float, default=0.0001)

args = parser.parse_args()

if(args.embeddings_model == "word2vec"):
    from gensim.models import Word2Vec
    w2vec =  Word2Vec.load(config.doc2vec_model_directory+"doc2vec_word2vec_dbow_hard_epoch9.model")
    embeddings_model = w2vec.wv
    # save memory
    del w2vec
else:
    import fasttext
    embeddings_model = fasttext.load_model(config.fasttext_model_directory+"/model.bin")

print("setting up datasource")

tr_gen, tr_steps, tr_y = get_preprocessed_generator(config.training_dataset_file, embeddings_model, config.embeddings_size, 5000, batch_size)
val_gen, val_steps, val_y = get_preprocessed_generator(config.validation_dataset_file, embeddings_model, config.embeddings_size, 5000, batch_size)
#te_gen, te_steps = get_preprocessed_generator(config.test_dataset_file, embeddings_model, config.embeddings_size, config.maxlen, batch_size)


input =  Input(shape=(None,config.embeddings_size), dtype='float32')

conv_4 = Conv1D(config.nr_filters*3,
                4,
                padding='same',
                activation='relu',
                strides=1)(input)
pooled = MaxPooling1D(pool_size=4, strides=None, padding='same')(conv_4)
shared = Model(input, pooled)

pr_1 =  Input(shape=(None, config.embeddings_size), dtype='float32')
pr_2 =  Input(shape=(None, config.embeddings_size), dtype='float32')

out_1 = shared(pr_1)
out_2 = shared(pr_2)

attention = AttentionLayer()([out_1,out_2])

# out_1 column wise

att_1 = GlobalMaxPooling1D()(attention)
att_1 = Activation('softmax')(att_1)
out_1 = dot([att_1, out_1], axes=1)

# out_2 row wise
attention_transposed = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(attention)
att_2 = GlobalMaxPooling1D()(attention_transposed)
att_2 = Activation('softmax')(att_2)
out_2 = dot([att_2, out_2], axes=1)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([out_1, out_2])

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

model = Model(input=[pr_1, pr_2], output=distance)

optimizer = Adam(lr=args.learning_rate)

model.compile(loss=contrastive_loss,
              optimizer=optimizer, metrics=[acc])


print('Train...')

checkpoint = ModelCheckpoint(config._current_path+"/classifier_models/cnn_attention_euclidian_"+args.embeddings_model+"/{val_loss:5.5f}.hdf5", monitor="val_loss", save_best_only=True)
early_stopping = EarlyStopping(monitor="val_loss", patience=config.early_stopping_patience)
csv_logger = CSVLogger(config._current_path+"/classifier_models/cnn_attention_euclidian_"+args.embeddings_model+"/training.csv")

model.fit_generator(tr_gen, steps_per_epoch=tr_steps,
                    epochs=epochs,
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    workers=1, callbacks=[checkpoint, early_stopping, csv_logger])
