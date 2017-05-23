from sklearn.svm import LinearSVC

from DuplicatePRs import config
from DuplicatePRs.dataset import load_csv, line_to_diff_files, read_normal


def filter_file_lines(str):
    #only keep file lines
    lines = str.split("\n")
    results = []
    for line in lines:
        if line[:10] == "diff --git":
            results.append(line)
    return results

def file_line_to_file(line):
    # remove diff --git + space
    line = line[12:]
    a,b = line.split(" ")
    return a

def file_to_files(file):
    pr_lines = filter_file_lines(read_normal(file))
    return map(file_line_to_file, pr_lines)

def files_to_percentages(prs1, prs2):
    results = []
    for i, pr in enumerate(prs1):
        pr1 = file_to_files(pr)
        pr2 = file_to_files(prs2[i])
        intersection = len([x for x in pr1 if x in pr2])
        total = (len(pr1)+len(pr2))*1.0
        results.append(intersection/total)
    return results



tr_1, tr_2, tr_y = map(line_to_diff_files, load_csv(config.training_dataset_file))
val_1, val_2, val_y = map(line_to_diff_files,load_csv(config.validation_dataset_file))
te_1, te_2, te_y = map(line_to_diff_files, load_csv(config.test_dataset_file))

tr = files_to_percentages(tr_1, tr_2)
val = files_to_percentages(val_1, val_2)
te = files_to_percentages(te_1, te_2)

svm = LinearSVC(verbose=1, max_iter=10000)
svm.fit(tr, tr_y)
acc = svm.score(val, val_y)






