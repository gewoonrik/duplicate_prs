import argparse

import struct
from keras.models import Model
from keras.models import load_model

from DuplicatePRs import config
from DuplicatePRs.dataset import load_csv, get_doc2vec_data_diffs

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='')

args = parser.parse_args()

model = load_model(config._current_path+"/classifier_models/doc2vec/"+args.model)

input = model.input

new_output =  model.layers[-2].output

model = Model(input, new_output)

test = load_csv(config.test_dataset_file)
test_1, test_2, test_labels = get_doc2vec_data_diffs(test)


predictions = model.predict([test_1, test_2])

label_out_file = config._current_path+"/doc2vec/embeddings/labels"
tensor_out_file = config._current_path+"/doc2vec/embeddings/tensors"
config_out_file = config._current_path+"/doc2vec/embeddings/projector_config.ptxt"



label_out = open(label_out_file, "w")
str = "owner\trepo\tduplicate\n"
for line in test:
    owner, repo, pr1,pr2,is_duplicate = line.split(",")
    str += owner+"\t"+repo+"\t"+is_duplicate+"\n"

label_out.write(str)
label_out.close()

tensor_out = open(tensor_out_file, 'wb')

for p in predictions:
    floatvals = p.tolist()
    for f in floatvals:
        tensor_out.write(struct.pack('<f', f))
tensor_out.close()

config_str = '''{
  "embeddings": [
    {
      "tensorName": "Test-dataset",
      "tensorShape": [%d, %d],
      "tensorPath": "%s",
      "metadataPath": "%s"
    }
  ],
  "modelCheckpointPath": "Demo datasets"
}''' % ( len(test), 600, label_out_file, tensor_out_file)



