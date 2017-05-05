import argparse

from keras.optimizers import Adam

from DuplicatePRs import config
from DuplicatePRs.classifiers.keras_abcnn import ABCNN
from DuplicatePRs.classifiers.preprocessing import get_preprocessed_generator

# Training
batch_size = 100
epochs = 150

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

model = ABCNN(
    left_seq_len=config.maxlen, right_seq_len=config.maxlen, depth=1,
    embed_dimensions=config.embeddings_size, nb_filter=config.nr_filters, filter_widths=3,
    collect_sentence_representations=False, abcnn_1=True, abcnn_2=False,
    mode="euclidean"
    # mode="cos"
)

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
