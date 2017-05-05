import pickle
import random
import numpy as np

def load_csv(file):
    f = open(file, "r")
    lines = f.read().split("\n")
    f.close()
    #remove head and empty line at bottom
    return lines[1:len(lines)-1]


def get_diff_file(owner,repo,id):
    return  "diff/"+owner+"@"+repo+"@"+id+".diff"

def get_tokenized_file(owner, repo, id):
    return  "diffs_tokenized/"+owner+"@"+repo+"@"+id+".diff"

def get_doc2vec_file(owner, repo, id):
    return  "diffs_doc2vec_preprocessed/"+owner+"@"+repo+"@"+id+".diff"


def line_to_diff_files(line):
    owner, repo, pr1, pr2, is_dup = line.split(",")
    pr1 = get_diff_file(owner,repo,pr1)
    pr2 = get_diff_file(owner,repo,pr2)
    return pr1, pr2

def _get_files(lines, get_file_func):
    files = []
    for line in lines:
        owner, repo, pr1, pr2, is_dup = line.split(",")
        pr1_file = get_file_func(owner,repo,pr1)
        pr2_file = get_file_func(owner,repo,pr2)
        files.append(pr1_file)
        files.append(pr2_file)
    return files

def get_tokenized_files(lines):
    return _get_files(lines, get_tokenized_file)

def get_diff_files(lines):
    return _get_files(lines, get_diff_file)

def read_pickled(file):
    f = open(file, "r")
    content = pickle.load(f)
    f.close()
    return content

def read_normal(file):
    f = open(file, "r")
    content = f.read()
    f.close()
    return content

def _get_data(lines, get_file_func, read):
    pr1s = []
    pr2s = []
    labels = []
    for line in lines:
        owner, repo, pr1, pr2, is_dup = line.split(",")
        pr1_file = get_file_func(owner, repo, pr1)
        pr2_file = get_file_func(owner, repo, pr2)

        pr1_data = read(pr1_file)
        pr2_data = read(pr2_file)

        pr1s.append(pr1_data)
        pr2s.append(pr2_data)
        labels.append(is_dup)

    return np.asarray(pr1s), np.asarray(pr2s), np.asarray(labels)


def get_doc2vec_data(lines):
    return _get_data(lines, get_doc2vec_file, read_pickled)



def get_tokenized_data(lines):
    return _get_data(lines, get_tokenized_file, read_pickled)