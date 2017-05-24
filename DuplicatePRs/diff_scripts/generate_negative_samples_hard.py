import traceback
from multiprocessing import Pool

import math
from tqdm import tqdm
from DuplicatePRs import config
from DuplicatePRs.file_baseline.diffs_to_files import get_overlapping_file_percentage, string_to_files
from filter_diffs import is_valid_string, is_valid_diff
from DuplicatePRs.dataset import get_diff_file, load_csv
from DuplicatePRs.diff_scripts.download import download_diff, download_diff_string

from pymongo import MongoClient
import random

project_cache = {}
def get_random_prs(db, owner,repo):
    prs = list(db.pull_requests.find({"owner":owner, "repo":repo}, {"number":1, "_id":0}).limit(5000))
    random.shuffle(prs)
    return prs


# keep trying getting random prs until they are valid and overlapping in at leas one file
def get_valid_random_prs_and_download(db, owner, repo):
    prs = get_random_prs(db, owner, repo)
    dict = {}
    has_valid = False
    for pr in prs:
        number = pr["number"]
        diff = download_diff_string(owner,repo,number)
        if is_valid_string(diff):
            has_valid = True
            files_in_diff = string_to_files(diff)
            for file in files_in_diff:
                if file in dict:
                    return number, dict[file]
                else:
                    dict[file] = number
    # if none found until now, just return two
    if not has_valid:
        print("no valid diffs found...")
        return 0,0
    print("no overlapping diffs found")
    results = []
    for pr in prs:
        number = pr["number"]
        diff = download_diff_string(owner,repo,number)
        if is_valid_string(diff):
            results.append(number)
        if len(results) == 2:
            return results[0], results[1]
    return prs[0]["number"], prs[1]["number"]


def generate_negative_sample(db, line):
    owner, repo, pr1, pr2 = line.split(",")

    rand1, rand2 = get_valid_random_prs_and_download(db, owner, repo)
    try:
        download_diff(owner,repo,rand1)
        download_diff(owner,repo,rand2)
    except:
        pass
    min_v = min(rand1, rand2)
    max_v = max(rand1, rand2)
    return owner, repo, str(min_v), str(max_v)

def batch_generate_negative_sample(batch):
    try:

        db = MongoClient('127.0.0.1', 27017).github
        results = []
        for line in tqdm(batch, total = len(batch)):
            results.append(generate_negative_sample(db, line))
        print("batch done :D")
        return results
    except Exception as e:
        print('Caught exception in worker thread (x = %d):' % x)

        # This prints the type, value, and stack trace of the
        # current exception being handled.
        traceback.print_exc()

        print()
        exit()

def batch(l, batches):
    per_batch = int(math.ceil(len(l)/(batches*1.0)))
    results = []
    for i in range(batches):
        results.append(l[i*per_batch:i*per_batch+per_batch])
    return results


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

    processes = 300
    p = Pool(processes)
    batched = batch(lines_filtered, processes)
    count = processes
    for b in p.imap_unordered(batch_generate_negative_sample, batched):
        count -=1
        print(str(count)+" threads to go!")
        i = 0
        for owner, repo, pr1, pr2 in b:
            f.write(owner+","+repo+","+pr1+","+pr2+","+"0\n")
            if i % 500 == 0:
                f.flush()
            i +=1
    f.close()


if __name__ == "__main__":
    generate_negative_samples(config._current_path+"/training.csv")
    generate_negative_samples(config._current_path+"/validation.csv")
    #generate_negative_samples(config._current_path+"/test.csv")

