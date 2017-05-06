import sys

from DuplicatePRs import config

reload(sys)
sys.setdefaultencoding('utf8')
import os.path
import pickle
from multiprocessing import Pool
from dataset import load_csv, get_diff_files, read_normal
from tokenize import tokenize,filter_diff_lines

training = load_csv(config.training_dataset_file)
validation = load_csv(config.validation_dataset_file)
test = load_csv(config.test_dataset_file)

total = training+test+validation

files = get_diff_files(total)

def tokenize_file(file):
    out_file = file.replace("diffs","diffs_tokenized")

    if not os.path.isfile(out_file):
        content = read_normal(file)
        tokens = tokenize(filter_diff_lines(content))
        f = open(out_file, "w")
        pickle.dump(tokens,f)
        f.close()

p = Pool(16)
p.map(tokenize_file,files)

