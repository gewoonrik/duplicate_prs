import argparse
import pickle
from multiprocessing import Pool

from gensim.models import Doc2Vec
from DuplicatePRs.dataset import load_csv, get_tokenized_files, read_pickled, get_tokenized_title_files, \
    get_tokenized_description_files
from DuplicatePRs import config

parser = argparse.ArgumentParser()
parser.add_argument('--titles', action="store_true")
parser.add_argument('--descriptions', action="store_true")

args = parser.parse_args()


if args.titles:
    get_files = get_tokenized_title_files
    file = "doc2vec_word2vec_dbow_title_epoch"
elif args.descriptions:
    get_files = get_tokenized_description_files
    print("learning based on descriptions")
    file = "doc2vec_word2vec_dbow_description_epoch"
else:
    get_files = get_tokenized_files
    file = "doc2vec_word2vec_dbow_epoch"


model = Doc2Vec.load(config._current_path+"/doc2vec_models/"+file+"9.model")

def docs2vec(file):
    content = read_pickled(file)
    vec = model.infer_vector(content)

    processed_pr = file.replace("_tokenized","_doc2vec_preprocessed")

    with open(processed_pr, 'w') as f:
        pickle.dump(vec, f)


tr_files = get_files(load_csv(config.training_dataset_file))
val_files = get_files(load_csv(config.validation_dataset_file))
te_files  = get_files(load_csv(config.test_dataset_file))

total = tr_files+val_files+te_files


p = Pool(16)
p.map(docs2vec,total)

