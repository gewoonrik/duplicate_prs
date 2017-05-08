#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import codecs

from multiprocessing import Pool

from pymongo import MongoClient

from DuplicatePRs import config
from DuplicatePRs.dataset import load_csv, get_title_file, get_tokenized_title_file

training = load_csv(config.training_dataset_file)
validation = load_csv(config.validation_dataset_file)
test = load_csv(config.test_dataset_file)

all = training+validation+test

client = MongoClient('127.0.0.1', 27017)


def get_title(owner, repo, id):
    db = client.github
    return db.pull_requests.find_one({"owner":owner, "repo":repo, "number": int(id)})["title"]

def get_and_save_title(owner, repo, id):
    title = get_title(owner,repo,id)
    title_file = get_title_file(owner, repo, id)
    f = codecs.open(title_file, "w", "utf-8")
    f.write(title)
    f.close()
    return title

def process_line(line):
    owner, repo, pr1, pr2, is_dup = line.split(",")
    title_1 = get_and_save_title(owner,repo,pr1)
    title_2 = get_and_save_title(owner,repo,pr2)

    tokenize_and_save_title(owner, repo, pr1, title_1)
    tokenize_and_save_title(owner, repo, pr2, title_2)

def tokenize_title(title):
    # keep the #, because that can sign for an issue or a PR
    preprocessed = ''.join(e for e in title if e.isalnum() or e == '#').lower()
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
    return [_f for _f in seq if _f]

def tokenize_and_save_title(owner, repo, id, title):
    file = get_tokenized_title_file(owner, repo, id)
    tokenized = tokenize_title(title)

    f = open(file, "w")
    pickle.dump(tokenized, f)
    f.close()



p = Pool(6)
p.map(process_line, all)








