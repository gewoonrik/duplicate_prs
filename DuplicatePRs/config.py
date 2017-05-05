import os

_current_path = os.path.dirname(os.path.abspath(__file__))
training_dataset_file = _current_path+"/training_with_negative_samples.csv"
validation_dataset_file = _current_path+"/validation_with_negative_samples.csv"
test_dataset_file = _current_path+"/test_with_negative_samples.csv"

embeddings_size = 300
maxlen = 1500

#convolution
nr_filters = 150

doc2vec_model_directory = _current_path+"/doc2vec_models/"
fasttext_model_directory = _current_path+"/fasttext/"