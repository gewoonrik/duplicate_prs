from multiprocessing import Pool

import math
from tqdm import tqdm
from itertools import imap, islice, chain
from DuplicatePRs import config
from DuplicatePRs.file_baseline.diffs_to_files import get_overlapping_file_percentage, string_to_files
from filter_diffs import is_valid_string, is_valid_diff
from DuplicatePRs.dataset import get_diff_file, load_csv
from DuplicatePRs.diff_scripts.download import download_diff, download_diff_string

from pymongo import MongoClient
import random


def get_random_prs(db, owner,repo):
    prs = list(db.pull_requests.find({"owner":owner, "repo":repo}, {"number":1})[:1000])
    random.shuffle(prs)
    return prs


# keep trying getting random prs until they are valid and overlapping in at leas one file
def get_valid_random_prs_and_download(db, owner, repo):
    prs = get_random_prs(db, owner, repo)
    dict = {}
    for pr in prs:
        number = pr["number"]
        diff = download_diff_string(owner,repo,number)
        if is_valid_string(diff):
            files_in_diff = string_to_files(diff)
            for file in files_in_diff:
                if file in dict:
                    return number, dict[file]
                else:
                    dict[file] = number
    # if none found until now, just return two
    print("no overlapping diffs found")
    return prs[0]["number"], prs[1]["number"]


def generate_negative_sample(db, line):
    owner, repo, pr1, pr2 = line.split(",")

    rand1, rand2 = get_valid_random_prs_and_download(db, owner, repo)
    download_diff(owner,repo,rand1)
    download_diff(owner,repo,rand2)
    min_v = min(rand1, rand2)
    max_v = max(rand1, rand2)
    return owner, repo, str(min_v), str(max_v)

def batch_generate_negative_sample(batch):
    db = MongoClient('127.0.0.1', 27017).github
    results = []
    for line in tqdm(batch, total = len(batch)):
        results.append(generate_negative_sample(db, line))
    return results

def batch(iterable, size):
    sourceiter = iter(iterable)
    while True:
        batchiter = islice(sourceiter, size)
        yield chain([batchiter.next()], batchiter)

def generate_negative_samples(file):
    lines = load_csv(file)
    lines_filtered = []
    for line in lines:
        owner, repo, pr1, pr2 = line.split(",")
        if is_valid_diff(get_diff_file(owner,repo,pr1)) and is_valid_diff(get_diff_file(owner,repo,pr2)):
            lines_filtered.append(line)

    f = open(file.split(".")[0]+"_with_negative_samples_hard.csv", "w")
    f.write("owner,repo,pr1_id,p2_id,is_duplicate\n")

    for line in lines_filtered:
        f.write(line+","+"1\n")

    processes = 16.0
    p = Pool(int(processes))
    per_process = math.ceil(len(lines_filtered)/processes)
    batched = batch(lines_filtered, per_process)

    for b in p.imap_unordered(batch_generate_negative_sample, batched):
        for owner, repo, pr1, pr2 in b:
            f.write(owner+","+repo+","+pr1+","+pr2+","+"0\n")
    f.close()


if __name__ == "__main__":
    generate_negative_samples(config._current_path+"/training.csv")
    generate_negative_samples(config._current_path+"validation.csv")
    generate_negative_samples(config._current_path+"test.csv")

