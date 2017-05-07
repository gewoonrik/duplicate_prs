import argparse
import numpy as np
from keras.models import load_model
from DuplicatePRs import config
from DuplicatePRs.dataset import load_csv, get_doc2vec_data

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='')

args = parser.parse_args()

model = load_model(args.model)
lines = load_csv(config.test_dataset_file)

test_1, test_2, test_labels = get_doc2vec_data(lines)


results = model.predict([test_1, test_2], batch_size=100)

lines_np = np.asarray(lines)

false_negatives = lines_np[results == 0 and test_labels == '1']
false_positives = lines_np[results == 1 and test_labels == '0']

print("\n".join(false_negatives))
