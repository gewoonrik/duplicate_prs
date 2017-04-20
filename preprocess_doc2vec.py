import pickle
from multiprocessing import Pool

from gensim.models import Doc2Vec
from tokenize import tokenize,filter_diff_lines

def load_data(file):
    f = open(file, "r")
    lines = f.read().split("\n")
    f.close()
    lines = lines[1:len(lines)-1]
    #remove head and empty line at bottom
    return lines

def lines_to_files(lines):
    files = []
    for line in lines:
        owner, repo, pr1, pr2, is_dup = line.split(",")
        pr1 = "diffs/"+owner+"@"+repo+"@"+pr1+".diff"
        pr2 = "diffs/"+owner+"@"+repo+"@"+pr2+".diff"
        files.append((pr1,pr2,is_dup))
    return files

model =  Doc2Vec.load("doc2vec.model")

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

