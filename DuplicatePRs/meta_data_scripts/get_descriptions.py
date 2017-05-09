#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import codecs

from multiprocessing import Pool

from pymongo import MongoClient

from DuplicatePRs import config
from DuplicatePRs.dataset import load_csv, get_description_file, get_tokenized_description_file

training = load_csv(config.training_dataset_file)
validation = load_csv(config.validation_dataset_file)
test = load_csv(config.test_dataset_file)

all = training+validation+test

client = MongoClient('127.0.0.1', 27017)


def get_description(owner, repo, id):
    db = client.github
    return db.pull_requests.find_one({"owner":owner, "repo":repo, "number": int(id)})["body"]

def get_and_save_description(owner, repo, id):
    description = get_description(owner,repo,id)
    description_file = get_description_file(owner, repo, id)
    f = codecs.open(description_file, "w", "utf-8")
    f.write(description)
    f.close()
    return description

def process_line(line):
    owner, repo, pr1, pr2, is_dup = line.split(",")
    description_1 = get_and_save_description(owner, repo, pr1)
    description_2 = get_and_save_description(owner, repo, pr2)

    tokenize_and_save_description(owner, repo, pr1, description_1)
    tokenize_and_save_description(owner, repo, pr2, description_2)

def tokenize_description(description):
    # keep the #, because that can sign for an issue or a PR
    preprocessed = ''.join(e for e in description if e.isalnum() or e == '#' or e.isspace()).lower()
    seq = []
    curr = ""
    # group all alpha characters, remove spaces/newlines
    # tokenize each number as word and tokenize # as one word
    for c in preprocessed:
        if c.isalpha():
            curr += c
        else:
            if curr != "":
                seq.append(curr)
                curr = ""
            if not c.isspace() or c == '\n':
                seq.append(c)
    if curr != "":
        seq.append(curr)
    return [_f for _f in seq if _f]

def tokenize_and_save_description(owner, repo, id, description):
    file = get_tokenized_description_file(owner, repo, id)
    tokenized = tokenize_description(description)

    f = open(file, "w")
    pickle.dump(tokenized, f)
    f.close()



p = Pool(6)
p.map(process_line, all)








