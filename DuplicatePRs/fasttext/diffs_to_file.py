#concats al diffs to one big file, so fasttext can process it
import pickle
import codecs

from DuplicatePRs.dataset import load_csv, get_tokenized_files, read_pickled



def files_to_file(files, output_file):
    f = codecs.open(output_file, mode="w", encoding="utf8")
    for file in files:
        content = read_pickled(file)
        content = [x.decode('utf-8','ignore') for x in content]
        content = " ".join(content)
        f.write(content)
    f.close()

training = get_tokenized_files(load_csv("training_with_negative_samples.csv"))
validation = get_tokenized_files(load_csv("validation_with_negative_samples.csv"))

files = training + validation

files_to_file(files, "fasttext/training_data.txt")

