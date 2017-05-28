import urllib

from flask import Flask
from flask import redirect
from flask import render_template
from flask import request
import re
import numpy as np
from gensim.models import Doc2Vec
from keras.models import load_model
from DuplicatePRs import config

from flask import url_for

from DuplicatePRs.tokenize import filter_diff_lines
from DuplicatePRs.tokenize import tokenize

app = Flask(__name__)

d2vec = Doc2Vec.load(config._current_path+"/doc2vec_models/doc2vec_word2vec_dbow_hard_epoch9.model")

model = load_model(config._current_path+"/classifier_models/doc2vec_hard/0.52447.hdf5")

@app.route("/")
def index():
    return render_template('select_diffs.html')

if __name__ == "__main__":
    app.run()

def check_url(url):

    return re.match('https://github.com/(.+)/(.+)/pull/([0-9]+)', url)

def get_diff(url):
    return urllib.urlopen(url).read()


@app.route('/predict', methods=['POST'])
def predict():
    pr1 = request.form['pr1']
    pr2 = request.form['pr1']
    if not check_url(pr1) or not check_url(pr2):
        return redirect(url_for('index'), code=400)
    pr1 = pr1+".diff"
    pr2 = pr2+".diff"

    pr1_diff = get_diff(pr1)
    pr2_diff = get_diff(pr2)

    inputs_1 = []
    inputs_2 = []


    vec1 = d2vec.infer_vector(tokenize(filter_diff_lines(pr1_diff)))
    vec2 = d2vec.infer_vector(tokenize(filter_diff_lines(pr2_diff)))
    inputs_1.append(vec1)
    inputs_2.append(vec2)

    result = model.predict([np.asarray(inputs_1), np.asarray(inputs_2)])
    return render_template('result.html', result=result)

