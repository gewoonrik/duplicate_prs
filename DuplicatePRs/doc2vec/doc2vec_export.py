from __future__ import print_function

#export all doc2vec data, so we can import them into the google projector
from DuplicatePRs import config
from DuplicatePRs.dataset import load_csv, get_doc2vec_data_diffs, get_doc2vec_file_diff, read_pickled

test = load_csv(config.test_dataset_file)
training = load_csv(config.training_dataset_file)
validation = load_csv(config.validation_dataset_file)


total = training+test+validation
length = len(total)
label_out_file = config._current_path+"/doc2vec/docs/labels"
tensor_out_file = config._current_path+"/doc2vec/docs/tensors"


label_out = open(label_out_file, "w")
tensor_out = open(tensor_out_file, 'wb')

label_out.write("owner\trepo\tid\n")
print("exporting")
for i, line in enumerate(test):
    owner, repo, pr1,pr2,is_duplicate = line.split(",")
    pr1_d = '\t'.join(map(str,read_pickled(get_doc2vec_file_diff(owner,repo,pr1)).tolist()))
    pr2_d = '\t'.join(map(str,read_pickled(get_doc2vec_file_diff(owner,repo,pr2)).tolist()))
    tensor_out.write(pr1_d+"\n")
    tensor_out.write(pr2_d+"\n")

    label_out.write(owner+"\t"+repo+"\t"+pr1+"\n")
    label_out.write(owner+"\t"+repo+"\t"+pr2+"\n")
    print("exporting %s / %s" % (i, length), end='\r')



label_out.close()
tensor_out.close()

