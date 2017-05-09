import argparse
import logging
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from DuplicatePRs.dataset import load_csv, get_tokenized_files, read_pickled, get_tokenized_title_files, \
    get_tokenized_description_files
from DuplicatePRs import config
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--titles', action="store_true")
parser.add_argument('--descriptions', action="store_true")

args = parser.parse_args()
cached_files = {}
class Documents(object):
    def __init__(self, files):
        self.files = files

    def __iter__(self):
        for file in self.files:
            if file in cached_files:
                content = cached_files[file]
            else:
                content = read_pickled(file)
                cached_files[file] = content
            yield TaggedDocument(words = content, tags = [file])


training = load_csv(config.training_dataset_file)
validation = load_csv(config.validation_dataset_file)

total = training+validation


if(args.titles):
    documents = Documents(get_tokenized_title_files(total))
    print("learning based on titles")
    file = "doc2vec_word2vec_dbow_title_epoch"
elif args.descriptions:
    documents = Documents(get_tokenized_description_files(total))
    print("learning based on descriptions")
    file = "doc2vec_word2vec_dbow_description_epoch"
else:
    documents = Documents(get_tokenized_files(total))
    print("learning based on diffs")
    file = "doc2vec_word2vec_dbow_epoch"


#iter = 1, because we keep training ourselves :)
model = Doc2Vec(size=config.embeddings_size, dbow_words= 1, dm=0, iter=1,  window=5, seed=1337, min_count=5, workers=16,alpha=0.025, min_alpha=0.025)
model.build_vocab(documents)
for epoch in range(10):
    print("epoch "+str(epoch))
    model.train(documents, total_examples=len(total)*2, epochs=1)
    model.save(config._current_path+'/doc2vec_models/'+file+str(epoch)+'.model')
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay




