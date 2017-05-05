from multiprocessing import Pool
from DuplicatePRs.diff_scripts.download import download_diff


def get_and_save_diff(line):
    # ignore last new line
    if "," in line:
        owner, repo, pr1, pr2 = line.split(",")
        download_diff(owner, repo, pr1)
        download_diff(owner, repo, pr2)


p = Pool(16)

f = open("temp/pairs.csv", "r")
lines = f.read().split("\n")
f.close()

p.map(get_and_save_diff, lines)
