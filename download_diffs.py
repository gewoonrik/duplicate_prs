import os
from multiprocessing import Pool
import urllib
dir = "diffs"


def download_diff(owner, repo, id):
    file = dir+"/"+owner+"@"+repo+"@"+id+".diff"

    if not os.path.isfile(file):
        url = "https://www.github.com/"+owner+"/"+repo+"/pull/"+id+".diff"
        urllib.urlretrieve(url, file)


def get_and_save_diff(line):
    # ignore last new line
    if "," in line:
        owner, repo, pr1, pr2 = line.split(",")
	download_diff(owner, repo, pr1)
        download_diff(owner, repo, pr2)


p = Pool(16)

f = open("pairs.csv", "r")
lines = f.read().split("\n")
f.close()

p.map(get_and_save_diff, lines)



