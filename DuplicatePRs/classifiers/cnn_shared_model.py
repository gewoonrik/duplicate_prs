from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Input, merge
from keras.models import Model
from keras.regularizers import l2

from DuplicatePRs import config


# First, define the vision modules
input =  Input(shape=(None,config.embeddings_size), dtype='float32')
conv_3 = Conv1D(config.nr_filters,
                3,
                padding='same',
                activation='relu',
                kernel_regularizer=l2(0.00001),
                bias_regularizer=l2(0.00001),
                strides=1)(input)
out_3 = GlobalMaxPooling1D()(conv_3)

conv_4 = Conv1D(config.nr_filters,
                4,
                padding='same',
                activation='relu',
                kernel_regularizer=l2(0.00001),
                bias_regularizer=l2(0.00001),
                strides=1)(input)
out_4 = GlobalMaxPooling1D()(conv_4)


conv_5 = Conv1D(config.nr_filters,
                5,
                padding='same',
                kernel_regularizer=l2(0.00001),
                bias_regularizer=l2(0.00001),
                activation='relu',
                strides=1)(input)
out_5 = GlobalMaxPooling1D()(conv_5)

output = merge([out_3, out_4, out_5], mode='concat')
conv_model = Model(input, output)