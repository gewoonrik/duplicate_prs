import argparse

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.layers import Input, merge
from keras.optimizers import Adam

from DuplicatePRs import config
from DuplicatePRs.classifiers.preprocessing import get_preprocessed_generator
from cnn_shared_model import conv_model

# Training
batch_size = 100
epochs = 150

parser = argparse.ArgumentParser()
parser.add_argument('--embeddings_model', default='word2vec')
parser.add_argument('--learning_rate', type=float, default=0.003)

args = parser.parse_args()

if(args.embeddings_model == "word2vec"):
    from gensim.models import Word2Vec
    embeddings_model =  Word2Vec.load("doc2vec_models/doc2vec_dbow_epoch9_notest.model")
    embeddings_model = embeddings_model.wv
else:
    import fasttext
    embeddings_model = fasttext.load_model("fasttext/model.bin")

tr_gen, tr_steps = get_preprocessed_generator(config.training_dataset_file, embeddings_model, config.embeddings_size, config.maxlen, batch_size)
val_gen, val_steps = get_preprocessed_generator(config.validation_dataset_file, embeddings_model, config.embeddings_size, config.maxlen, batch_size)
te_gen, te_steps = get_preprocessed_generator(config.test_dataset_file, embeddings_model, config.embeddings_size, config.maxlen, batch_size)


print('Build model...')


pr_1 =  Input(shape=(config.maxlen, config.embeddings_size), dtype='float32')
pr_2 =  Input(shape=(config.maxlen, config.embeddings_size), dtype='float32')

out_1 = conv_model(pr_1)
out_2 = conv_model(pr_2)

merged = merge([out_1, out_2], mode='concat')
x = Dropout(0.2)(merged)
x = Dense(600, activation='relu')(x)
x = Dense(600, activation='relu')(x)
main_output = Dense(1, activation='sigmoid', name='main_output')(x)

model = Model(input=[pr_1, pr_2], output=main_output)

optimizer = Adam(lr=args.learning_rate)

model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

print('Train...')
model.fit_generator(tr_gen, steps_per_epoch=tr_steps,
                    epochs=epochs,
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    workers=1)
score, acc = model.evaluate_generator(te_gen, steps=te_steps)
print('Test score:', score)
print('Test accuracy:', acc)
