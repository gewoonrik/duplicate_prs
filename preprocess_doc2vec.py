import pickle
from multiprocessing import Pool

from gensim.models import Doc2Vec


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

def read(file):
    f = open(file, "r")
    content = f.read()
    f.close()
    return content

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
    lines = lines[1:len(lines)-1]
    #remove head and empty line at bottom
    return lines

def lines_to_files(lines):
    files = []
    for line in lines:
        owner, repo, pr1, pr2, is_dup = line.split(",")
        pr1 = "diffs/"+owner+"@"+repo+"@"+pr1+".diff"
        pr2 = "diffs/"+owner+"@"+repo+"@"+pr2+".diff"
        files.append((pr1,pr2,is_dup))
    return files

model =  Doc2Vec.load("doc2vec.model")

def docs2vec(file):
    (pr1, pr2, is_dup) = file
    content_1 = read(pr1)
    content_2 = read(pr2)
    vec1 = model.infer_vector(tokenize(filter_diff_lines(content_1)))
    vec2 = model.infer_vector(tokenize(filter_diff_lines(content_2)))

    processed_pr1 = pr1.replace("diffs","diffs_doc2vec_preprocessed")
    processed_pr2 = pr2.replace("diffs","diffs_doc2vec_preprocessed")

    with open(processed_pr1, 'w') as f:
        pickle.dump(vec1, f)
    with open(processed_pr2, 'w') as f:
        pickle.dump(vec2, f)


training = lines_to_files(load_data("training_with_negative_samples.csv"))
validation = lines_to_files(load_data("validation_with_negative_samples.csv"))
test = lines_to_files(load_data("test_with_negative_samples.csv"))

total = training+validation+test


p = Pool(16)
p.map(docs2vec,total)

