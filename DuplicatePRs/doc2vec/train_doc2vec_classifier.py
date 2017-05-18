from keras.callbacks import CSVLogger, EarlyStopping, TensorBoard
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
x = Dense(600, activation='relu', name="dense_1")(x)
main_output = Dense(1, activation='sigmoid', name='output')(x)

model = Model(input=[pr1, pr2], output=[main_output])

optimizer = Adam(lr = 0.0001)

str = "owner\trepo\tduplicate\n"
for line in train:
    owner, repo, pr1,pr2,is_duplicate = line.split(",")
    str += owner+"\t"+repo+"\t"+is_duplicate+"\n"

f = open(config._current_path+"/classifier_models/doc2vec2/embeddings/meta.tsv", "w")
f.write(str)
f.close()

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint(config._current_path+"/classifier_models/doc2vec2/{val_loss:5.5f}.hdf5", monitor="val_loss", save_best_only=True)
early_stopping = EarlyStopping(monitor="val_loss", patience=config.early_stopping_patience)
csv_logger = CSVLogger(config._current_path+"/classifier_models/doc2vec2/training.csv")
tsb = TensorBoard(log_dir=config._current_path+"/classifier_models/doc2vec2/embeddings",
                            write_images=True,
                            embeddings_freq=1,
                            embeddings_layer_names=['dense_1'],
                            batch_size=5,
                            embeddings_metadata= config._current_path+"/classifier_models/doc2vec2/embeddings/meta.tsv")


print("train")


model.fit([tr_1, tr_2], tr_labels, batch_size=100, nb_epoch=100,
          shuffle=False,
          validation_data=([val_1, val_2], val_labels), callbacks=[checkpoint, early_stopping, csv_logger, tsb])

results = model.evaluate([test_1, test_2], test_labels, batch_size=100)
print('Test results: ', results)
print('On metrics: ', model.metrics_names)
