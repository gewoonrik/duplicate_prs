import urllib

from flask import Flask
from flask import redirect
from flask import render_template
from flask import request
import re
import numpy as np
from gensim.models import Doc2Vec
from keras.models import load_model, model_from_json
from DuplicatePRs import config
import keras.backend as K
from flask import url_for

from DuplicatePRs.classifiers.preprocessing import preprocess
from DuplicatePRs.tokenize import filter_diff_lines
from DuplicatePRs.tokenize import tokenize
from DuplicatePRs.visualisation.cam import test_lines, test_lines_word2vec
from DuplicatePRs.visualisation.visualize import visualize

app = Flask(__name__)

d2vec = Doc2Vec.load(config._current_path+"/doc2vec_models/doc2vec_word2vec_dbow_hard_epoch9.model")
embeddings_model = d2vec.wv

model = load_model(config._current_path+"/classifier_models/doc2vec_hard/0.50974.hdf5")

top_model = load_model(config._current_path+"/classifier_models/word2vec2doc_hard/0.51685.hdf5")
def acc(y_true, y_pred):
    ones = K.ones_like(y_pred)
    return K.mean(K.equal(y_true, ones - K.clip(K.round(y_pred), 0, 1)), axis=-1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    # duplicates should be low, non duplicates should be high
    # so duplicates = 0, non duplicate = 1
    y_true = -1 * y_true + 1
    return K.mean((1 - y_true) * K.square(y_pred) +  y_true * K.square(K.maximum(margin - y_pred, 0)))

f = open(config._current_path+"/classifier_models/cnn_euclidian/model.json")
json = f.read()
f.close()
shared_model = model_from_json(json, {"contrastive_loss":contrastive_loss, "acc":acc})
shared_model.load_weights(config._current_path+"/classifier_models/cnn_euclidian_word2vec_hard/best.hdf5")

# take only the shared CNN model :)
shared_model = shared_model.layers[-2]

def pair_to_word2vec(embeddings_model, tok1, tok2):
    p1 = preprocess([tok1],embeddings_model, 300, len(tok1))
    p2 = preprocess([tok2],embeddings_model, 300, len(tok2))
    return p1, p2

@app.route("/")
def index():
    return render_template('select_diffs.html', to='predict')

@app.route("/w2vec")
def w2vec():
    return render_template('select_diffs.html', to='predict_w2vec')

if __name__ == "__main__":
    app.run()

def check_url(url):

    return re.match('https://github.com/(.+)/(.+)/pull/([0-9]+)', url)

def get_diff(url):
    return urllib.urlopen(url).read()

def to_style(tokens, res):
    styled = []
    for i,token in enumerate(tokens):
        if token == 'lineremovedtoken' or token == 'lineaddedtoken':
            styled.append("<br/>")
        styled.append("<span style='rgba(0, 255, 0, "+str(res[i])+")'>"+token+"</span>")
    return ' '.join(styled)

@app.route('/predict_w2vec', methods=['POST'])
def predict_w2vec():
    pr1 = request.form['pr1']
    pr2 = request.form['pr2']
    if not check_url(pr1) or not check_url(pr2):
        return redirect(url_for('w2vec'), code=400)
    pr1 = pr1+".diff"
    pr2 = pr2+".diff"

    pr1_diff = get_diff(pr1)
    pr2_diff = get_diff(pr2)
    vec1 = tokenize(filter_diff_lines(pr1_diff))
    vec2 = tokenize(filter_diff_lines(pr2_diff))
    w2vec1, w2vec2 = pair_to_word2vec(embeddings_model, vec1, vec2)

    results = visualize(shared_model, top_model, w2vec1, w2vec2)
    #s1 = to_style(vec1, results[0])
    #s2 = to_style(vec2, results[1])
    return render_template('side_by_side.html', pr1_tokens=vec1, pr2_tokens=vec2, res1=results[0], res2=results[1])


@app.route("/w2vec_cam")
def w2vec_cam():
    return render_template('select_diffs.html', to='predict_w2vec_cam')


@app.route('/predict_w2vec_cam', methods=['POST'])
def predict_w2vec_cam():
    pr1 = request.form['pr1']
    pr2 = request.form['pr2']
    if not check_url(pr1) or not check_url(pr2):
        return redirect(url_for('w2vec'), code=400)
    pr1 = pr1+".diff"
    pr2 = pr2+".diff"

    pr1_diff = get_diff(pr1)
    pr2_diff = get_diff(pr2)
    vec1 = tokenize(filter_diff_lines(pr1_diff))
    vec2 = tokenize(filter_diff_lines(pr2_diff))

    pred1, pred2, result  = test_lines_word2vec(embeddings_model, shared_model, top_model, vec1, vec2)

    # only keep the lines that reduce the result when removed :)
    influence1 = np.power(-1 * np.minimum(pred1, 0), 2)
    influence2 = np.power(-1 * np.minimum(pred2, 0), 2)
    sum1 = np.max(influence1)
    sum2 = np.max(influence2)
    influence1 = influence1/sum1
    influence2 = influence2/sum2


    bad_influence1 = np.maximum(pred1, 0)
    bad_influence2 = np.maximum(pred2, 0)
    sum1 = bad_influence1.sum()
    sum2 = bad_influence2.sum()
    bad_influence1 = bad_influence1/sum1
    bad_influence2 = bad_influence2/sum2
    return render_template('side_by_side_lines.html', pr1_diff = pr1_diff.decode('utf-8', 'ignore').split("\n"), pr2_diff = pr2_diff.decode('utf-8', 'ignore').split("\n"),
                       influence1 = influence1, influence2 = influence2, bad_influence1= bad_influence1, bad_influence2=bad_influence2)




@app.route('/predict', methods=['POST'])
def predict():
    pr1 = request.form['pr1']
    pr2 = request.form['pr2']
    if not check_url(pr1) or not check_url(pr2):
        return redirect(url_for('index'), code=400)
    pr1 = pr1+".diff"
    pr2 = pr2+".diff"

    pr1_diff = get_diff(pr1)
    pr2_diff = get_diff(pr2)


    tokenized_1 = tokenize(filter_diff_lines(pr1_diff))
    tokenized_2 = tokenize(filter_diff_lines(pr2_diff))

    pred1, pred2, result  = test_lines(d2vec, model, tokenized_1, tokenized_2)

    # only keep the lines that reduce the result when removed :)
    influence1 = -1 * np.minimum(pred1, 0)
    influence2 = -1 * np.minimum(pred2, 0)
    sum1 = np.max(influence1)
    sum2 = np.max(influence2)
    influence1 = influence1/sum1
    influence2 = influence2/sum2


    bad_influence1 = np.maximum(pred1, 0)
    bad_influence2 = np.maximum(pred2, 0)
    sum1 = bad_influence1.sum()
    sum2 = bad_influence2.sum()
    bad_influence1 = bad_influence1*0#bad_influence1/sum1
    bad_influence2 = bad_influence2*0#/sum2

    print(result)
    return render_template('side_by_side_lines.html', pr1_diff = pr1_diff.decode('utf-8', 'ignore').split("\n"), pr2_diff = pr2_diff.decode('utf-8', 'ignore').split("\n"),
                           influence1 = influence1, influence2 = influence2, bad_influence1= bad_influence1, bad_influence2=bad_influence2)

