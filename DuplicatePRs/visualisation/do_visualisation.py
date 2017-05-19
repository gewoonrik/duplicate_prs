from keras.models import load_model

from DuplicatePRs import config
from DuplicatePRs.classifiers.preprocessing import get_preprocessed_generator
from DuplicatePRs.visualisation.visualize import visualize

best_model = "0.16909.hdf5"
model = load_model(config._current_path+"/classifier_models/cnn_euclidian/"+best_model)

from gensim.models import Word2Vec
w2vec =  Word2Vec.load(config.doc2vec_model_directory+"doc2vec_word2vec_dbow_epoch9.model")
embeddings_model = w2vec.wv
# save memory
del w2vec

gen, steps, labels = get_preprocessed_generator(config.test_dataset_file, embeddings_model, config.embeddings_size, config.maxlen, 1)

pr1, pr2, label = gen[0]

print(visualize(model, pr1))

