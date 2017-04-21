import pickle
from multiprocessing import Pool

from gensim.models import Doc2Vec
from tokenize import tokenize,filter_diff_lines
from load_data import load_data, lines_to_files

model =  Doc2Vec.load("doc2vec.model")

def read(file):
    f = open(file, "r")
    content = f.read()
    f.close()
    return content

def docs2vec(file):
    (pr1, pr2, is_dup) = file
    content_1 = read(pr1)
    content_2 = read(pr2)
    vec1 = model.infer_vector(tokenize(filter_diff_lines(content_1)))
    vec2 = model.infer_vector(tokenize(filter_diff_lines(content_2)))

    processed_pr1 = pr1.replace("diffs","diffs_doc2vec_preprocessed")
    processed_pr2 = pr2.replace("diffs","diffs_doc2vec_preprocessed")

    with open(processed_pr1, 'w') as f:
        pickle.dump(vec1, f)
    with open(processed_pr2, 'w') as f:
        pickle.dump(vec2, f)


training = lines_to_files(load_data("training_with_negative_samples.csv"))
validation = lines_to_files(load_data("validation_with_negative_samples.csv"))
test = lines_to_files(load_data("test_with_negative_samples.csv"))

total = training+validation+test


p = Pool(16)
p.map(docs2vec,total)

