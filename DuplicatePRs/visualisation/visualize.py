from keras import backend as K
import numpy as np
from keras.models import Model

def get_activations(model, inputs):
    activations = []
    inp = model.input
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    if len(inputs.shape) == 3:
        batch_inputs = inputs[np.newaxis, ...]
    else:
        batch_inputs = inputs
    layer_outputs = [func([batch_inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        print(layer_activations)
    return activations



def visualize(model, pr):
    """

    :type model: Model
    """

    shared_model = model.layers[-2]
    activations = get_activations(shared_model, pr)

    dim = calculate_iim(pr, activations, shared_model)
    return dim


def calculate_iim(inputs, activations, model):
    conv3 = model.layers[1]
    conv4 = model.layers[2]
    conv5 = model.layers[3]

    conv3_out = activations[0]
    conv4_out = activations[1]
    conv5_out = activations[2]

    max3_out = activations[3]
    max4_out = activations[4]
    max5_out = activations[5]


    iim = np.ones(activations[7].shape())
    iim_merges = calc_iim_merge(iim, 3)
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
    max = np.max(arr)*1.0
    return arr/max

def calc_iim_merge(iim, filter_count):
    size = len(iim)
    per_filter = size/filter_count
    return [np.asarray(iim[0+i*per_filter: 0+i*per_filter+per_filter]) for i in range(filter_count)]

def calc_iim_max_pooling(iim, input, output):
    # input is (sequence_length, filters)
    # output is (filters)
    iim_out = np.zeros((input.shape[1],input.shape[0]))
    for i, inp in enumerate(input):
        for f_nr, feat in enumerate(inp):
            if output[f_nr] == input[i][f_nr]:
                iim_out[f_nr][i] = iim[f_nr]
    return iim_out

def calc_iim_conv(iim, input, layer):
    iim_out = np.zeros(input.shape)

    weights = layer.get_weights()[0]
    filter_length = weights.shape[0]
    embedding_size = weights.shape[1]
    nr_filters = weights.shape[2]
    input_length = len(input)
    # input is (sequence_length, embeddings)
    # output is (sequence_length, filters)

    for i in range(input_length):
        for j in range(embedding_size):
            sum = 0
            for k in range(nr_filters):
                for l in range(input_length + 1 - filter_length):
                    influence = iim[k][l]
                    inp = input[i][j]
                    index = i+1-l
                    if index >= 0 and index < filter_length:
                        sum += influence * inp * weights[index][j][k]
            iim_out[i][j] = sum

    return iim_out
