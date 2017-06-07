import sys

from DuplicatePRs import config
from DuplicatePRs.diff_scripts.download import download_diff

reload(sys)
sys.setdefaultencoding('utf8')
import os.path
import pickle
from multiprocessing import Pool
from dataset import load_csv, get_diff_files, read_normal
from tokenize import tokenize,filter_diff_lines
from tqdm import tqdm


training = load_csv(config.training_dataset_file)
validation = load_csv(config.validation_dataset_file)
test = load_csv(config.test_dataset_file)

total = training+validation + test

files = get_diff_files(total)


def download_a_diff(line):
    owner, repo, id1, id2, is_dup = line.split(",")
    download_diff(owner,repo,id1)
    download_diff(owner,repo,id2)

def tokenize_file(file):
    out_file = file.replace("diffs","diffs_tokenized")

    if not os.path.isfile(out_file):
        content = read_normal(file)
        tokens = tokenize(filter_diff_lines(content))
        f = open(out_file, "w")
        pickle.dump(tokens,f)
        f.close()

p = Pool(150)
#p.map(download_a_diff, total)
for i in  tqdm(p.imap_unordered(tokenize_file,files), total = len(files)):
    pass


