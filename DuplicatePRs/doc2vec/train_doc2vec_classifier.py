from keras.callbacks import CSVLogger
from keras.layers import Input, merge, Dense, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from DuplicatePRs.dataset import load_csv, get_doc2vec_data_diffs, get_doc2vec_data_titles
from keras.optimizers import Adam
from DuplicatePRs import config

print("loading files")
train = load_csv(config.training_dataset_file)
validation = load_csv(config.validation_dataset_file)
test = load_csv(config.test_dataset_file)

tr_1, tr_2, tr_labels = get_doc2vec_data_diffs(train)
val_1, val_2, val_labels = get_doc2vec_data_diffs(validation)
test_1, test_2, test_labels = get_doc2vec_data_diffs(test)

tr_titles_1, tr_titles_2, _ = get_doc2vec_data_titles(train)
val_titles_1, val_titles_2, _ = get_doc2vec_data_titles(validation)
te_titles_1, te_titles_2, _ = get_doc2vec_data_titles(test)

#pr1 = Input(shape=(300,), dtype='float32', name='pr1_input')
title1 = Input(shape=(300,), dtype='float32', name='title1_input')

#pr2 = Input(shape=(300,), dtype='float32', name='pr2_input')
title2 = Input(shape=(300,), dtype='float32', name='title2_input')

#p1 = merge([pr1, title1])
#p1 = Dense(300, activation='relu')(p1)

#p2 = merge([pr2, title2])
@p2 = Dense(300, activation='relu')(p2)

x = merged = merge([title1, title2], mode='concat')
x = Dense(600, activation='relu')(x)
main_output = Dense(1, activation='sigmoid', name='output')(x)

model = Model(input=[title1, title2], output=[main_output])

optimizer = Adam(lr = 0.00011)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint(config._current_path+"/classifier_models/doc2vec_classifier-{val_acc:5.5f}.hdf5", monitor="val_acc", save_best_only=True)


print("train")


model.fit([tr_titles_1, tr_titles_2], tr_labels, batch_size=100, nb_epoch=100,
          validation_data=([val_titles_1, val_titles_2], val_labels), callbacks=[checkpoint])

results = model.evaluate([te_titles_1, te_titles_2], test_labels, batch_size=100)
print('Test results: ', results)
print('On metrics: ', model.metrics_names)
