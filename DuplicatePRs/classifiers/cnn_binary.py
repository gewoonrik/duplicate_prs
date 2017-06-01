import argparse

from keras.callbacks import EarlyStopping, CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Dense,Dropout
from keras.layers import Input, merge
from keras.optimizers import Adam

from DuplicatePRs import config
from DuplicatePRs.classifiers.preprocessing import get_preprocessed_generator
from cnn_shared_model import conv_model

# Training
batch_size = 50
epochs = 150

parser = argparse.ArgumentParser()
parser.add_argument('--embeddings_model', default='word2vec')
parser.add_argument('--learning_rate', type=float, default=0.0001)

args = parser.parse_args()

if(args.embeddings_model == "word2vec"):
    from gensim.models import Word2Vec
    embeddings_model =  Word2Vec.load(config.doc2vec_model_directory+"doc2vec_word2vec_dbow_hard_epoch9.model")
    embeddings_model = embeddings_model.wv
else:
    import fasttext
    embeddings_model = fasttext.load_model(config.fasttext_model_directory+"/model.bin")


tr_gen, tr_steps, tr_y = get_preprocessed_generator(config.training_dataset_file, embeddings_model, config.embeddings_size, config.maxlen, batch_size)
val_gen, val_steps, val_y = get_preprocessed_generator(config.validation_dataset_file, embeddings_model, config.embeddings_size, config.maxlen, batch_size)
#te_gen, te_steps, te_y = get_preprocessed_generator(config.test_dataset_file, embeddings_model, config.embeddings_size, 10000, batch_size)


print('Build model...')


pr_1 =  Input(shape=(None, config.embeddings_size), dtype='float32')
pr_2 =  Input(shape=(None, config.embeddings_size), dtype='float32')

out_1 = conv_model(pr_1)
out_2 = conv_model(pr_2)

merged = merge([out_1, out_2], mode='concat')
x = Dense(2000, activation='relu')(merged)
x = Dropout(0.5)
main_output = Dense(1, activation='sigmoid', name='main_output')(x)

model = Model(input=[pr_1, pr_2], output=main_output)

optimizer = Adam(lr=args.learning_rate)

model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])


print('Train...')

checkpoint = ModelCheckpoint(config._current_path+"/classifier_models/cnn_binary_"+args.embeddings_model+"_hard/{val_loss:5.5f}.hdf5", monitor="val_loss", save_best_only=True)
early_stopping = EarlyStopping(monitor="val_loss", patience=config.early_stopping_patience)
csv_logger = CSVLogger(config._current_path+"/classifier_models/cnn_binary_"+args.embeddings_model+"_hard/training.csv")

model.fit_generator(tr_gen, steps_per_epoch=tr_steps,
                    epochs=epochs,
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    workers=1,  callbacks=[checkpoint, early_stopping, csv_logger])
#score, acc = model.evaluate_generator(te_gen, steps=te_steps)
#print('Test score:', score)
#print('Test accuracy:', acc)
