import logging
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from DuplicatePRs.dataset import load_csv, get_tokenized_files, read_pickled
from DuplicatePRs import config
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Documents(object):
    def __init__(self, files):
        self.files = files

    def __iter__(self):
        for file in self.files:
            content = read_pickled(file)
            yield TaggedDocument(words = content, tags = [file])


training = load_csv(config.training_dataset_file)
validation = load_csv(config.validation_dataset_file)

total = training+validation


documents = Documents(get_tokenized_files(total))

model = Doc2Vec(size=config.embeddings_size, dm=0,  window=5, seed=1337, min_count=5, workers=16,alpha=0.025, min_alpha=0.025)
model.build_vocab(documents)
for epoch in range(10):
    print("epoch "+str(epoch))
    model.train(documents)
    model.save(config._current_path+'/doc2vec_models/doc2vec_dbow_epoch'+str(epoch)+'.model')
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay




