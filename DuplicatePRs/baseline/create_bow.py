from gensim.corpora import Dictionary

from DuplicatePRs import config
from DuplicatePRs.dataset import get_tokenized_data, load_csv

tpr1s, tpr2s, _ = get_tokenized_data(load_csv(config.training_dataset_file))
valpr1s, valpr2s, _ = get_tokenized_data(load_csv(config.validation_dataset_file))

print("creating dictionairy")
dict = Dictionary(tpr1s + tpr2s + valpr1s + valpr2s)
print("filtering")
dict.filter_extremes(keep_n = None)
print("saving")
dict.save(config._current_path+"/baseline/dict")