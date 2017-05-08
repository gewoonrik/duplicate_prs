import pickle
from multiprocessing import Pool

from gensim.models import Doc2Vec
from DuplicatePRs.dataset import load_csv, get_tokenized_files, read_pickled
from DuplicatePRs import config

model = Doc2Vec.load(config._current_path+"/doc2vec_models/doc2vec_dbow_epoch.model")

def docs2vec(file):
    content = read_pickled(file)
    vec = model.infer_vector(content)

    processed_pr = file.replace("diffs_tokenized","diffs_doc2vec_preprocessed")

    with open(processed_pr, 'w') as f:
        pickle.dump(vec, f)


tr_files = get_tokenized_files(load_csv(config.training_dataset_file))
val_files = get_tokenized_files(load_csv(config.validation_dataset_file))
te_files  = get_tokenized_files(load_csv(config.test_dataset_file))

total = tr_files+val_files+te_files


p = Pool(16)
p.map(docs2vec,total)

