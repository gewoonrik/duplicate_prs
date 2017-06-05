from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
def to_lines(tokens):
    lines = []
    cur_line = []
    for token in tokens:
        if token == "lineremovedtoken" or token == "lineaddedtoken" or token == "newfiletoken":
            lines.append(cur_line)
            cur_line = []
        cur_line.append(token)
    if len(cur_line) > 0:
        lines.append(cur_line)
    return lines


def check_line(doc2vec, lines, i):
    before = lines[:i]
    after = lines[i+1:]
    test = [x for sublist in (before + after) for x in sublist]
    vec = doc2vec.infer_vector(test)
    return vec

def get_predictions(doc2vec, model, baseline, lines, other_vector, first):
    results = []
    print("go")
    for i in tqdm(range(len(lines))):
        res = check_line(doc2vec, lines, i)
        if first:
            results[i] = model.predict([np.asarray([res]), np.asarray([other_vector])])[0][0] - baseline
        else:
            results[i] = model.predict([np.asarray([other_vector]), np.asarray([res])])[0][0] - baseline
    return results

def test_lines(doc2vec, model, pr1, pr2):
    print("to lines")
    lines1 = to_lines(pr1)
    lines2 = to_lines(pr2)
    print("get base vectors")
    vec1 = doc2vec.infer_vector(pr1)
    vec2 = doc2vec.infer_vector(pr2)
    print("baseline")
    baseline = model.predict([np.asarray([vec1]), np.asarray([vec2])])[0][0]
    predictions1 = get_predictions(doc2vec, model, baseline, lines1, vec2, True)
    predictions2 = get_predictions(doc2vec, model, baseline, lines2, vec1, False)
    return np.asarray(predictions1), np.asarray(predictions2), baseline
