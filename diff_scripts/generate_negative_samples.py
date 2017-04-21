import os
import urllib
from multiprocessing import Pool


from pymongo import MongoClient
import random



def download_diff(owner, repo, id):
    file = "diffs/"+owner+"@"+repo+"@"+str(id)+".diff"

    if not os.path.isfile(file):
        url = "https://www.github.com/"+owner+"/"+repo+"/pull/"+str(id)+".diff"
        urllib.urlretrieve(url, file)

def get_random_pr(owner, repo):
    client = MongoClient('127.0.0.1', 27017)
    db = client.github
    count = db.pull_requests.count({"owner":owner, "repo":repo})
    rand = random.randint(0, count-1)
    return db.pull_requests.find({"owner":owner, "repo":repo})[rand]

def generate_negative_sample(line):
    owner, repo, pr1, pr2 = line.split(",")
    print(owner+" "+repo)
    rand1 = get_random_pr(owner, repo)
    download_diff(owner,repo,rand1["number"])

    rand2 = get_random_pr(owner, repo)
    download_diff(owner,repo,rand2["number"])

    min_v = min(rand1["number"], rand2["number"])
    max_v = max(rand1["number"], rand2["number"])
    return owner, repo, str(min_v), str(max_v)

def generate_negative_samples(file):
    f = open(file, "r")
    lines = f.read().split("\n")
    f.close()
    
    #remove header and last newline
    lines = lines[1:len(lines)-1]

    f = open(file.split(".")[0]+"_with_negative_samples.csv", "w")
    f.write("owner,repo,pr1_id,p2_id,is_duplicate\n")
    for line in lines:
        f.write(line+","+"1\n")

    p = Pool(10)
    negative_samples = p.map(generate_negative_sample, lines)
    
    for owner, repo, pr1, pr2 in negative_samples:
        f.write(owner+","+repo+","+pr1+","+pr2+","+"0\n")
    f.close()

generate_negative_samples("training.csv")
generate_negative_samples("validation.csv")
generate_negative_samples("test.csv")

