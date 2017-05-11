from keras.callbacks import CSVLogger, EarlyStopping
from keras.layers import Input, merge, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from DuplicatePRs.dataset import load_csv, get_doc2vec_data_diffs
from DuplicatePRs import config
from keras.optimizers import Adam
from DuplicatePRs import config

print("loading files")
train = load_csv(config.training_dataset_file)
validation = load_csv(config.validation_dataset_file)
test = load_csv(config.test_dataset_file)
tr_1, tr_2, tr_labels = get_doc2vec_data_diffs(train)
val_1, val_2, val_labels = get_doc2vec_data_diffs(validation)
test_1, test_2, test_labels = get_doc2vec_data_diffs(test)

pr1 = Input(shape=(300,), dtype='float32', name='pr1_input')
pr2 = Input(shape=(300,), dtype='float32', name='pr2_input')

x = merged = merge([pr1, pr2], mode='concat')
x = Dense(600, activation='relu')(x)
main_output = Dense(1, activation='sigmoid', name='output')(x)

model = Model(input=[pr1, pr2], output=[main_output])

optimizer = Adam(lr = 0.0001)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint(config._current_path+"/classifier_models/doc2vec/doc2vec_classifier-{val_acc:5.5f}.hdf5", monitor="val_acc", save_best_only=True)
early_stopping = EarlyStopping(monitor="val_loss", patience=config.early_stopping_patience)
csv_logger = CSVLogger(config._current_path+"/classifier_models/doc2vec/doc2vec_classifier.csv")


print("train")


model.fit([tr_1, tr_2], tr_labels, batch_size=100, nb_epoch=100,
          validation_data=([val_1, val_2], val_labels), callbacks=[checkpoint, early_stopping, csv_logger])

results = model.evaluate([test_1, test_2], test_labels, batch_size=100)
print('Test results: ', results)
print('On metrics: ', model.metrics_names)
