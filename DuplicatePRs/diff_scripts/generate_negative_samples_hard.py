from multiprocessing import Pool

from tqdm import tqdm

from DuplicatePRs import config
from DuplicatePRs.file_baseline.diffs_to_files import get_overlapping_file_percentage, file_to_files
from filter_diffs import is_valid_diff
from DuplicatePRs.dataset import get_diff_file, load_csv
from DuplicatePRs.diff_scripts.download import download_diff


from pymongo import MongoClient
import random


def get_random_pr(owner, repo):
    client = MongoClient('127.0.0.1', 27017)
    db = client.github
    count = db.pull_requests.count({"owner":owner, "repo":repo})
    rand = random.randint(0, count-1)
    return db.pull_requests.find({"owner":owner, "repo":repo})[rand]

def get_random_prs(owner,repo):
    client = MongoClient('127.0.0.1', 27017)
    db = client.github
    prs = db.pull_requests.find({"owner":owner, "repo":repo}, {"number":1})[:1000]
    random.shuffle(prs)
    return prs


# keep trying getting random prs until they are valid and overlapping in at leas one file
def get_valid_random_prs_and_download(owner, repo):
    prs = get_random_prs(owner, repo)
    dict = {}
    for pr in prs:
        number = pr["number"]
        diff = download_diff(owner,repo,number)
        if is_valid_diff(diff):
            files_in_diff = file_to_files(diff)
            for file in files_in_diff:
                if file in dict:
                    return number, dict[file]
                else:
                    dict[file] = number
    # if none found until now, just return two
    print("no overlapping diffs found")
    return prs[0]["number"], prs[1]["number"]


def generate_negative_sample(line):
    owner, repo, pr1, pr2 = line.split(",")
    tries = 0
    # try to get overlapping diffs for 100 times
    # else settle with a non overlapping diff
    rand1, rand2 = get_valid_random_prs_and_download(owner, repo)
    min_v = min(rand1, rand2)
    max_v = max(rand1, rand2)
    return owner, repo, str(min_v), str(max_v)

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

    p = Pool(10)
    for owner, repo, pr1, pr2 in tqdm(p.imap_unordered(generate_negative_sample, lines_filtered)):
        f.write(owner+","+repo+","+pr1+","+pr2+","+"0\n")
    f.close()


if __name__ == "__main__":
    generate_negative_samples(config._current_path+"/training.csv")
    generate_negative_samples(config._current_path+"validation.csv")
    generate_negative_samples(config._current_path+"test.csv")

