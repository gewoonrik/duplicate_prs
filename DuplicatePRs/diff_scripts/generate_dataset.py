import sys
from pymongo import MongoClient
import re
from os import listdir
from os.path import isfile, join


dir = "comments"

files = [f for f in listdir(dir) if isfile(join(dir, f))]
nr_files = len(files)
keywords = ["favour", "favor", "resubmission", "continuation", "superset", "dup", "dupe", "duplicate"]

client = MongoClient('127.0.0.1', 27017)
db = client.github

count = 0
def progressBar(value, endvalue, bar_length=20):

    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stderr.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stderr.flush()

# tries to find a PR reference in this comment
# returns a pair of PRs when one is found
def find_pr_reference(file_name, str):
    owner, repo, issue_id, rest = file_name.split("@")

    matched_ids = re.findall("#([0-9]+)", str)
    for matched_id in matched_ids:
        #filter > INT32
        if int(matched_id) < 2147483647:
            res = db.pull_requests.find_one({"repo": repo, "owner": owner, "number":int(matched_id)})
            if res != None:
                return owner, repo, issue_id, int(matched_id)
    return False

# find all comments that contain one of the keywords in `keywords`
# then searches the PR the comment is refering to.
def search_match(file):
    global count
    progressBar(count, nr_files)
    count += 1
    f = open(dir+"/"+file, 'r')
    str = f.read()
    f.close()

    # remove files that contain more than 4 lines, because they are probably not PR references.
    if str.count("\n") > 4:
        return False

    for keyword in keywords:
        if " "+keyword+" " in str:
            return find_pr_reference(file, str)

    return False

results = [search_match(file) for file in files]


resulting_pairs = [item for item in results if item != False]

f = open("temp/pairs.csv", 'w')
f.write("owner,repo,pr1_id,p2_id\n")

for owner, repo, pr1_id, pr2_id in resulting_pairs:
    if pr1_id != pr2_id:
        min_v = min(pr1_id, pr2_id)
        max_v = max(pr1_id, pr2_id)
        f.write(owner+","+repo+","+str(min_v)+","+str(max_v)+"\n")
f.close()







