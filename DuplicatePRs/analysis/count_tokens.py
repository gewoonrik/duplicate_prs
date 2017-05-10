# counts the number of tokens of the PRs

from DuplicatePRs.dataset import load_csv, _current_path, get_tokenized_data
from DuplicatePRs import config

training = load_csv(config.training_dataset_file)
validation = load_csv(config.validation_dataset_file)
test = load_csv(config.test_dataset_file)
all = training + validation + test

files = get_tokenized_data(all)


nr_tokens = map(len, files)

f = open(_current_path+"/analysis/nrtokens.txt", "w")
txt = "\n".join(str(x) for x in nr_tokens)
f.write(txt)
f.close()