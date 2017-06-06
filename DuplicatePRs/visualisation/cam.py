from tqdm import tqdm
import numpy as np

from DuplicatePRs.classifiers.preprocessing import preprocess


def to_lines(tokens):
    lines = []
    cur_line = []
    for token in tokens:
        if len(cur_line) >0 and (token == "lineremovedtoken" or token == "lineaddedtoken" or token == "newfiletoken"):
            lines.append(cur_line)
            cur_line = []
        cur_line.append(token)
    if len(cur_line) > 0:
        lines.append(cur_line)
    return lines

lines_per_check = 5

def check_line(doc2vec, lines, i):
    before = lines[:i]
    after = lines[i+lines_per_check:]
    test = [x for sublist in (before + after) for x in sublist]
    vec = get_doc2vec(doc2vec, test, 10)
    return vec


def get_doc2vec(doc2vec, pr, sample_count):
    sum = np.zeros(300)
    for i in range(sample_count):
        sum += doc2vec.infer_vector(pr)
    return sum/sample_count

def get_predictions(doc2vec, model, baseline, lines, other_vector, first):
    results = np.zeros(len(lines))
    print("go")
    for i in tqdm(range(len(lines))):
        res = check_line(doc2vec, lines, i)
        if first:
            res = model.predict([np.asarray([res]), np.asarray([other_vector])])[0][0] - baseline
        else:
            res = model.predict([np.asarray([other_vector]), np.asarray([res])])[0][0] - baseline
        for j in range(lines_per_check):
            if i+j < len(lines):
                results[i+j] += res
    return results

def test_lines(doc2vec, model, pr1, pr2):
    print("to lines")
    lines1 = to_lines(pr1)
    lines2 = to_lines(pr2)
    print("get base vectors")
    vec1 = get_doc2vec(doc2vec, pr1, 10)
    vec2 = get_doc2vec(doc2vec, pr2, 10)
    print("baseline")
    baseline = model.predict([np.asarray([vec1]), np.asarray([vec2])])[0][0]
    predictions1 = get_predictions(doc2vec, model, baseline, lines1, vec2, True)
    predictions2 = get_predictions(doc2vec, model, baseline, lines2, vec1, False)
    return np.asarray(predictions1), np.asarray(predictions2), baseline







def pair_to_word2vec(embeddings_model, tok1, tok2):
    p1 = preprocess([tok1],embeddings_model, 300, len(tok1))
    p2 = preprocess([tok2],embeddings_model, 300, len(tok2))
    return p1, p2

def check_line_word2vec(shared_model, lines, i):
    # set the line we are checking to zeroes
    lines[i] = lines[i] * 0
    #flatten
    test = [x for sublist in lines for x in sublist]
    return shared_model.predict([np.asarray([test])])



def get_predictions_word2vec(shared_model, top_model, baseline, lines, other_vector, first):
    results = np.zeros(len(lines))
    print("go")
    for i in tqdm(range(len(lines))):
        res = check_line_word2vec(shared_model, lines, i)
        if first:
            res = top_model.predict([np.asarray(res), np.asarray(other_vector)])[0][0] - baseline
        else:
            res = top_model.predict([np.asarray(other_vector), np.asarray(res)])[0][0] - baseline
        results[i] += res
    return results

def w2vec2lines(lines, pr):
    pr = pr[0]
    w2vec_lines = []
    i = 0
    for line in lines:
        w2vec_lines.append(pr[i:len(line)+1])
        i = len(line)
    return w2vec_lines

def test_lines_word2vec(word2vec, shared_model, top_model, pr1, pr2):
    w2vec1, w2vec2 = pair_to_word2vec(word2vec, pr1, pr2)

    lines1 = to_lines(pr1)
    lines2 = to_lines(pr2)

    # returns wrapped in a list
    base_1 = shared_model.predict([w2vec1])
    base_2 = shared_model.predict([w2vec2])

    baseline = top_model.predict([base_1, base_2])[0][0]


    w2vec1_lines = w2vec2lines(lines1, w2vec1)
    w2vec2_lines = w2vec2lines(lines2, w2vec2)
    # cleanup memory, we need it
    del w2vec1
    del w2vec2

    pred1 = get_predictions_word2vec(shared_model, top_model, baseline, w2vec1_lines, base_2, True)
    pred2 = get_predictions_word2vec(shared_model, top_model, baseline, w2vec2_lines, base_1, False)

    return np.asarray(pred1), np.asarray(pred2), baseline