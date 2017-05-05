# removes all lines from pairs of which a diff could not be downloaded
import os.path
from DuplicatePRs.dataset import load_csv, get_diff_file


def diff_exists(owner, repo, id):
    file = get_diff_file(owner,repo,id)
    val = os.path.isfile(file)
    if not val:
        print file
    return val


lines = load_csv("temp/pairs.csv")

f = open("temp/pairs.csv", "w")
f.write("owner,repo,pr1_id,p2_id\n")

for line in lines:
    owner, repo, pr1, pr2 = line.split(",")
    if diff_exists(owner, repo, pr1) and diff_exists(owner, repo, pr2):
        f.write(owner + "," + repo + "," + pr1 + "," + pr2 + "\n")

f.close()
