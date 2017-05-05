from keras.callbacks import CSVLogger
from keras.layers import Input, merge, Dense, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from DuplicatePRs.dataset import load_csv, get_doc2vec_data
from DuplicatePRs import config
from keras.optimizers import Adam


print("loading files")
tr_1, tr_2, tr_labels = get_doc2vec_data(load_csv(config.training_dataset_file))
val_1, val_2, val_labels = get_doc2vec_data(load_csv(config.validation_dataset_file))
test_1, test_2, test_labels = get_doc2vec_data(load_csv(config.test_dataset_file))

pr1 = Input(shape=(300,), dtype='float32', name='pr1_input')
pr2 = Input(shape=(300,), dtype='float32', name='pr2_input')
x = merged = merge([pr1, pr2], mode='concat')
x = Dropout(0.4)(x)
x = Dense(600, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(600, activation='relu')(x)
x = Dropout(0.4)(x)
main_output = Dense(1, activation='sigmoid', name='output')(x)

model = Model(input=[pr1, pr2], output=[main_output])

optimizer = Adam(lr = 0.00011)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint("classifier_models/doc2vec_classifier-{val_acc:5.5f}.hdf5", monitor="val_acc", save_best_only=True)


csv_logger = CSVLogger('trainingslog_doc2vec.csv')
# early_stopping = EarlyStopping(monitor='val_loss', patience=20)
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
#              patience=2, min_lr=0.00001)
print("train")


model.fit([tr_1, tr_2], tr_labels, batch_size=100, nb_epoch=100,
          validation_data=([val_1, val_2], val_labels), callbacks=[csv_logger])

results = model.evaluate([test_1, test_2], test_labels, batch_size=100)
print('Test results: ', results)
print('On metrics: ', model.metrics_names)
