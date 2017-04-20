from multiprocessing import Pool

from tokenize import tokenize,filter_diff_lines
from load_data import load_data, lines_to_files
import pickle

training = load_data("training_with_negative_samples.csv")
validation = load_data("validation_with_negative_samples.csv")
test = load_data("test_with_negative_samples.csv")

total = training+test+validation

files = lines_to_files(total)

def tokenize_file(file):
    f = open(file, "r")
    content = f.read()
    f.close()
    tokens = tokenize(filter_diff_lines(content))
    out_file = file.replace("diffs_tokenized")
    f = open(out_file, "w")
    pickle.dump(tokens,f)
    f.close()

p = Pool(16)
p.map(tokenize_file,total)