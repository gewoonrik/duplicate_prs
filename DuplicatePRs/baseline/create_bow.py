from gensim.corpora import Dictionary
from itertools import chain

from DuplicatePRs import config
from DuplicatePRs.dataset import load_csv, get_tokenized_data_generator

def get_prs(gen):
    for pr1,pr2,_ in gen:
        yield map(lambda x: x.decode('utf-8', 'ignore'), pr1)
        yield map(lambda x: x.decode('utf-8', 'ignore'), pr2)

tr_gen = get_tokenized_data_generator(load_csv(config.training_dataset_file))
val_gen = get_tokenized_data_generator(load_csv(config.validation_dataset_file))

both_gen = chain(tr_gen,val_gen)


print("creating dictionairy")
dict = Dictionary(get_prs(both_gen))
print("filtering")
dict.filter_extremes(no_below = 2, keep_n = None)
print("saving")
dict.save(config._current_path+"/baseline/dict_hard")