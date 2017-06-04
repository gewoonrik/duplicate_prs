from keras import backend as K
import numpy as np

def get_activations(model, inputs):
    activations = []
    inp = model.inputs
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        list_inputs = [inputs]
    else:
        list_inputs = []
        list_inputs.extend(inputs)
        list_inputs.append(0.)
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
    return activations





def visualize(shared_model, top_model, pr1, pr2):


    pr1_act = get_activations(shared_model, [pr1])
    pr2_act = get_activations(shared_model, [pr2])

    top_activations = get_activations(top_model, [[pr1_act[-1][0]], [pr2_act[-1][0]]])
    print("result ")
    print(top_activations[-1])
    top_iim = calculate_iim_top(top_activations, top_model)

    dim1 = calculate_iim_shared(top_iim[:300], pr1[0], pr1_act, shared_model)
    dim2 = calculate_iim_shared(top_iim[300:], pr2[0], pr2_act, shared_model)

    return dim1, dim2

def calculate_iim_top(activations, model):
    global test_iim
    start_iim = activations[-1]

    iim = calc_iim_dense(start_iim,model.layers[-1].get_weights()[0],activations[-2][0])
    # we skip dropout
    iim = calc_iim_dense(iim,model.layers[-3].get_weights()[0],activations[-4][0])
    return iim

def calculate_iim_shared(start_iim, inputs, activations, model):
    conv3 = model.layers[1]
    conv4 = model.layers[2]
    conv5 = model.layers[3]


    conv3_out = activations[1][0]
    conv4_out = activations[2][0]
    conv5_out = activations[3][0]

    max3_out = activations[4][0]
    max4_out = activations[5][0]
    max5_out = activations[6][0]

    print(inputs.shape)


    iim_merges = calc_iim_concat(start_iim, 3)
    iim_max3 = calc_iim_max_pooling(iim_merges[0], conv3_out, max3_out)
    iim_max4 = calc_iim_max_pooling(iim_merges[1], conv4_out, max4_out)
    iim_max5 = calc_iim_max_pooling(iim_merges[2], conv5_out, max5_out)

    iim_conv3 = calc_iim_conv(iim_max3, inputs, conv3)
    iim_conv4 = calc_iim_conv(iim_max4, inputs, conv4)
    iim_conv5 = calc_iim_conv(iim_max5, inputs, conv5)

    iim_sum = iim_conv3 + iim_conv4 + iim_conv5
    final = np.zeros(inputs.shape[0])
    for i in range(inputs.shape[0]):
        sum = 0
        for j in range(inputs.shape[1]):
            sum += iim_sum[i][j]
        final[i] = sum
    return normalize_nparr(final)

def normalize_nparr(arr):
    #remove less than zero's
    arr = np.maximum(arr, 0)
    #mi = np.min(arr)
    #arr = arr-mi
    ma = np.max(arr)*1.0
    return arr/ma

def calc_iim_dense(iim_vec, weights, input):
    iim_weights = (iim_vec* weights).transpose()

    assert (iim_weights.transpose())[0][0] == iim_vec[0]*weights[0][0]
    assert (iim_weights.transpose())[10][5] == iim_vec[10]*weights[10][5]

    return np.dot((input * iim_weights).transpose()
                  , np.repeat(1, weights.shape[1]))

def calc_iim_concat(iim, filter_count):
    size = iim.shape[0]
    per_filter = size/filter_count
    return [np.asarray(iim[0+i*per_filter: 0+i*per_filter+per_filter]) for i in range(filter_count)]

def calc_iim_max_pooling(iim, input, output):
    # iim.shape = (filters)
    # input is (sequence_length, filters)
    # output iim is (sequence, filters)
    iim_out = np.zeros((input.shape[0],input.shape[1]))
    for i, inp in enumerate(input):
        for f_nr, feat in enumerate(inp):
            if output[f_nr] == input[i][f_nr]:
                iim_out[i][f_nr] = iim[f_nr]
    return iim_out
from tqdm import tqdm

def calc(iim, weights, input_length, inp):
    (i, input) = inp
    filter_length = weights.shape[0]
    embedding_size = weights.shape[1]
    nr_filters = weights.shape[2]
    out = np.zeros(embedding_size)
    padding = filter_length//2
    for j in range(embedding_size):
        inp = input[j]
        sum = 0
        for k in range(nr_filters):
            for l in range(filter_length):
                index = i-l+padding
                if index<input_length and index > 0:
                    influence = iim[index][k]
                    sum += influence * weights[l][j][k]
        out[j] = inp * sum
    return i,out

def calc_iim_conv(iim, input, layer):
    # iim.shape = (sequence, filters)
    weights = layer.get_weights()[0]
    input_length = len(input)

    # input is (sequence_length, embeddings)
    # output is (sequence_length, embeddings)

    iim_out = np.zeros(input.shape)
    from functools import partial


    from multiprocessing import Pool
    p = Pool(16)
    func = partial(calc, iim, weights, input_length)
    for i,vec in tqdm(p.imap_unordered(func,enumerate(input)), total=input_length):
        iim_out[i] = vec
    return iim_out