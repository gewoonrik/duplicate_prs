import argparse
import pickle
from keras.models import load_model
import numpy as np

from DuplicatePRs.dataset import load_csv, get_doc2vec_data

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='')

args = parser.parse_args()

model = load_model(args.model)
lines = load_csv("test_with_negative_samples.csv")

test_1, test_2, test_labels = get_doc2vec_data(lines)


results = model.predict([test_1, test_2], batch_size=100)

false_negatives = lines[results == 0 and test_labels == 1]
print("\n".join(false_negatives))
