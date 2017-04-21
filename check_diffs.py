# removes all lines from pairs of which a diff could not be downloaded
import os.path

dir = "diffs"


def diff_exists(owner, repo, id):
    file = dir + "/" + owner + "@" + repo + "@" + id + ".diff"
    val = os.path.isfile(file)
    if not val:
        print file
    return val


f = open("pairs.csv", "r")
lines = f.read().split("\n")
f.close()
# remove header and newline at the end
lines = lines[1:len(lines) - 1]

f = open("pairs.csv", "w")
f.write("owner,repo,pr1_id,p2_id\n")

for line in lines:
    owner, repo, pr1, pr2 = line.split(",")
    if diff_exists(owner, repo, pr1) and diff_exists(owner, repo, pr2):
        f.write(owner + "," + repo + "," + pr1 + "," + pr2 + "\n")

f.close()
