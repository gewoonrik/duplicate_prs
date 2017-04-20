import logging
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from tokenize import tokenize,filter_diff_lines
from load_data import load_data, lines_to_files


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Documents(object):
    def __init__(self, files):
        self.files = files

    def __iter__(self):
        for file in self.files:
            f = open(file, "r")
            content = f.read()
            f.close()
            yield TaggedDocument(words = tokenize(filter_diff_lines(content)), tags = [file])


training = load_data("training_with_negative_samples.csv")
validation = load_data("validation_with_negative_samples.csv")
test = load_data("test_with_negative_samples.csv")

total = training+test+validation




documents = Documents(lines_to_files(total))

model = Doc2Vec(size=300, dm=0,  window=5, seed=1337, min_count=5, workers=16,alpha=0.025, min_alpha=0.025)
model.build_vocab(documents)
for epoch in range(10):
    print("epoch "+str(epoch))
    model.train(documents)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay
model.save('doc2vec_dbow10.model')


