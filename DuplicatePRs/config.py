import os

training_dataset_file = os.path.abspath(__file__)+"/training_with_negative_samples.csv"
validation_dataset_file = os.path.abspath(__file__)+"/validation_with_negative_samples.csv"
test_dataset_file = os.path.abspath(__file__)+"/test_with_negative_samples.csv"

embeddings_size = 300
maxlen = 1500

#convolution
nr_filters = 150