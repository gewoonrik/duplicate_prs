import logging
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

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

def filter_diff_lines(str):
    # remove everything that isn't an added or removed line.
    lines = str.split("\n")
    results = []
    for line in lines:
        length = len(line)
        if length > 0:
            if line[0] == "+" or line[0] == "-":
                if length == 1 or ( line[1] != "+" and line[1] != "-"):
                    if line[0] == "+":
                        line = "LINE__ADDED__TOKEN" + line[1:]
                    elif line[0] == "-":
                        line = "LINE__REMOVED__TOKEN" + line[1:]
                    results.append(line)
            elif line[:10] == "diff --git":
                results.append("NEW__FILE__TOKEN")
    return "\n".join(results)


def tokenize(text, lower=True):
    ''' Tokenizes code. All consecutive alphanumeric characters are grouped into one token.
    Thereby trying to heuristically match identifiers.
    All other symbols are seen as one token.
    Whitespace is stripped, except the newline token.
    '''
    if lower:
        text = text.lower() #type: str
    seq = []
    curr = ""
    for c in text:
        if c.isalnum():
            curr += c
        else:
            if curr != "":
                seq.append(curr)
                curr = ""
            if not c.isspace() or c == '\n':
                seq.append(c)
    return [_f for _f in seq if _f]


def load_data(file):
    f = open(file, "r")
    lines = f.read().split("\n")
    f.close()
    #remove head and empty line at bottom
    return lines[1:len(lines)-1]

def lines_to_files(lines):
    files = []
    for line in lines:
        owner, repo, pr1, pr2, is_dup = line.split(",")
        files.append("diffs/"+owner+"@"+repo+"@"+pr1+".diff")
        files.append("diffs/"+owner+"@"+repo+"@"+pr2+".diff")
    return files

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


