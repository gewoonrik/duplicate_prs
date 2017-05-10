# counts the number of tokens of the PRs
from multiprocessing import Pool

from DuplicatePRs.dataset import load_csv, _current_path, get_tokenized_files, read_pickled
from DuplicatePRs import config

training = load_csv(config.training_dataset_file)
validation = load_csv(config.validation_dataset_file)
test = load_csv(config.test_dataset_file)
all = training + validation + test


def count_tokens(file):
    content = read_pickled(file)
    return len(content)


files = get_tokenized_files(all)

p = Pool(4)

nr_tokens = p.map(count_tokens, files)

f = open(_current_path+"/analysis/nrtokens.txt", "w")
txt = "\n".join(str(x) for x in nr_tokens)
f.write(txt)
f.close()