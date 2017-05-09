import pickle
import random
import numpy as np
from functools import partial
from multiprocessing import Pool
import os

_current_path = os.path.dirname(os.path.abspath(__file__))


def load_csv(file):
    f = open(file, "r")
    lines = f.read().split("\n")
    f.close()
    #remove header
    lines = lines[1:len(lines)-1]
    random.shuffle(lines)
    return lines


def get_title_file(owner,repo,id):
    return  _current_path+"/titles/"+owner+"@"+repo+"@"+str(id)+".title"

def get_tokenized_title_file(owner,repo,id):
    return  _current_path+"/titles_tokenized/"+owner+"@"+repo+"@"+str(id)+".title"

def get_description_file(owner,repo,id):
    return  _current_path+"/descriptions/"+owner+"@"+repo+"@"+str(id)+".description"

def get_tokenized_description_file(owner,repo,id):
    return  _current_path+"/descriptions_tokenized/"+owner+"@"+repo+"@"+str(id)+".description"

def get_doc2vec_description_file(owner,repo,id):
    return  _current_path+"/descriptions_doc2vec_preprocessed/"+owner+"@"+repo+"@"+str(id)+".description"



def get_diff_file(owner,repo,id):
    return  _current_path+"/diffs/"+owner+"@"+repo+"@"+str(id)+".diff"

def get_tokenized_file(owner, repo, id):
    return  _current_path+"/diffs_tokenized/"+owner+"@"+repo+"@"+str(id)+".diff"

def get_doc2vec_file_diff(owner, repo, id):
    return  _current_path+"/diffs_doc2vec_preprocessed/"+owner+"@"+repo+"@"+str(id)+".diff"

def get_doc2vec_file_title(owner, repo, id):
    return  _current_path+"/titles_doc2vec_preprocessed/"+owner+"@"+repo+"@"+str(id)+".title"
def get_doc2vec_file_description(owner, repo, id):
    return  _current_path+"/descriptions_doc2vec_preprocessed/"+owner+"@"+repo+"@"+str(id)+".description"


def line_to_files(line, file_func):
    owner, repo, pr1, pr2, is_dup = line.split(",")
    pr1 = file_func(owner,repo,pr1)
    pr2 = file_func(owner,repo,pr2)
    return pr1, pr2, is_dup

def line_to_diff_files(line):
    return line_to_files(line, get_diff_file)

def line_to_tokenized_files(line):
    return line_to_files(line, get_tokenized_file)


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

def get_tokenized_title_files(lines):
    return _get_files(lines, get_tokenized_title_file)

def get_tokenized_description_files(lines):
    return _get_files(lines, get_tokenized_description_file)

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

def _get_data_by_line(get_file_func, read, maxlen, line):
    owner, repo, pr1, pr2, is_dup = line.split(",")
    pr1_file = get_file_func(owner, repo, pr1)
    pr2_file = get_file_func(owner, repo, pr2)

    pr1_data = read(pr1_file)
    pr2_data = read(pr2_file)
    if maxlen != None:
        pr1_data = pr1_data[:maxlen]
        pr2_data = pr2_data[:maxlen]
    return pr1_data, pr2_data, is_dup

def _get_data(lines, get_file_func, read, maxlen = None):
    get_data_func = partial(_get_data_by_line, get_file_func, read, maxlen)
    pool = Pool(8)

    data = pool.map(get_data_func, lines)
    pool.close()

    #http://stackoverflow.com/questions/7558908/unpacking-a-list-tuple-of-pairs-into-two-lists-tuples
    pr1s, pr2s, labels = zip(*data)
    return np.asarray(pr1s), np.asarray(pr2s), np.asarray(labels)


def get_doc2vec_data_diffs(lines):
    return _get_data(lines, get_doc2vec_file_diff, read_pickled, None)

def get_doc2vec_data_titles(lines):
    return _get_data(lines, get_doc2vec_file_title, read_pickled, None)

def get_doc2vec_data_descriptions(lines):
    return _get_data(lines, get_doc2vec_file_description, read_pickled, None)

def get_tokenized_data(lines, maxlen = None):
    return _get_data(lines, get_tokenized_file, read_pickled, maxlen)