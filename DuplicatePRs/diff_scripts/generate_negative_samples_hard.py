from multiprocessing import Pool

from DuplicatePRs import config
from DuplicatePRs.file_baseline.diffs_to_files import get_overlapping_files_percentage
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


# keep trying getting random prs until they are valid
def get_valid_random_pr_and_download(owner, repo):
    while True:
        rand = get_random_pr(owner, repo)
        file = download_diff(owner,repo,rand["number"])
        if is_valid_diff(file):
            return rand, file

count = 0
def generate_negative_sample(line):
    global count
    owner, repo, pr1, pr2 = line.split(",")
    print(owner+" "+repo)
    tries = 0
    # try to get overlapping diffs for 100 times
    # else settle with a non overlapping diff
    while tries < 100:
        rand1, file1 = get_valid_random_pr_and_download(owner,repo)
        rand2, file2 = get_valid_random_pr_and_download(owner,repo)
        if get_overlapping_files_percentage(file1, file2) > 0:
            break
        tries += 1
    count +=1
    print("count "+str(count))
    min_v = min(rand1["number"], rand2["number"])
    max_v = max(rand1["number"], rand2["number"])
    return owner, repo, str(min_v), str(max_v)

def generate_negative_samples(file):
    lines = load_csv(file)
    lines_filtered = []
    for line in lines:
        owner, repo, pr1, pr2 = line.split(",")
        if is_valid_diff(get_diff_file(owner,repo,pr1)) and is_valid_diff(get_diff_file(owner,repo,pr2)):
            lines_filtered.append(line)

    f = open(config._current_path+"/"+file.split(".")[0]+"_with_negative_samples_hard.csv", "w")
    f.write("owner,repo,pr1_id,p2_id,is_duplicate\n")

    for line in lines_filtered:
        f.write(line+","+"1\n")

    p = Pool(10)
    print("need to do "+str(len(lines_filtered)))
    negative_samples = p.map(generate_negative_sample, lines_filtered)

    for owner, repo, pr1, pr2 in negative_samples:
        f.write(owner+","+repo+","+pr1+","+pr2+","+"0\n")
    f.close()


if __name__ == "__main__":
    generate_negative_samples(config._current_path+"/training.csv")
    generate_negative_samples(config._current_path+"validation.csv")
    generate_negative_samples(config._current_path+"test.csv")

