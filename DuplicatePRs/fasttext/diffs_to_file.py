#concats al diffs to one big file, so fasttext can process it
import codecs

from DuplicatePRs import config
from DuplicatePRs.dataset import load_csv, get_tokenized_files, read_pickled



def files_to_file(files, output_file):
    f = codecs.open(output_file, mode="w", encoding="utf8")
    for file in files:
        content = read_pickled(file)
        content = [x.decode('utf-8','ignore') for x in content]
        content = " ".join(content)
        f.write(content)
    f.close()

training = get_tokenized_files(load_csv(config.training_dataset_file))
validation = get_tokenized_files(load_csv(config.validation_dataset_file))

files = training + validation

files_to_file(files, config._current_path+"/fasttext/training_data.txt")

