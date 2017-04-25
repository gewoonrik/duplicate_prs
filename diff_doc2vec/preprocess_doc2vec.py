import pickle
from multiprocessing import Pool

from gensim.models import Doc2Vec
from tokenize import tokenize,filter_diff_lines
from load_data import load_data

model =  Doc2Vec.load("doc2vec_dbow10_epoch4.model")

def line_to_data(line):
    owner, repo, pr1, pr2, is_dup = line.split(",")
    pr1 = "diffs/"+owner+"@"+repo+"@"+pr1+".diff"
    pr2 = "diffs/"+owner+"@"+repo+"@"+pr2+".diff"
    return pr1, pr2, is_dup

def lines_to_data(lines):
    l = []
    for line in lines:
        l.append(line_to_data(line))
    return l

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


training = lines_to_data(load_data("training_with_negative_samples2.csv"))
validation = lines_to_data(load_data("validation_with_negative_samples2.csv"))
test = lines_to_data(load_data("test_with_negative_samples2.csv"))

total = training+validation+test


p = Pool(16)
p.map(docs2vec,total)

