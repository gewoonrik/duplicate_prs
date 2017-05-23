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
    a = line.split(" b/")
    return a[0]

def file_to_files(file):
    pr_lines = filter_file_lines(read_normal(file))
    return map(file_line_to_file, pr_lines)

def string_to_files(string):
    pr_lines = filter_file_lines(read_normal(file))
    return map(file_line_to_file, pr_lines)

def get_overlapping_file_percentage(pr1, pr2):
    pr1 = file_to_files(pr1)
    pr2 = file_to_files(pr2)
    intersection = len([x for x in pr1 if x in pr2])
    total = (len(pr1)+len(pr2))*1.0
    if total == 0:
        return 1.0
    return intersection/total

def get_overlapping_files_percentage(prs1, prs2):
    results = []
    for i, pr in enumerate(prs1):
        results.append([get_overlapping_file_percentage(pr,prs2[i])])
    return results

if __name__ == "__main__":


    tr_1, tr_2, tr_y = zip(*map(line_to_diff_files, load_csv(config.training_dataset_file)))
    val_1, val_2, val_y = zip(*map(line_to_diff_files,load_csv(config.validation_dataset_file)))
    te_1, te_2, te_y = zip(*map(line_to_diff_files, load_csv(config.test_dataset_file)))

    tr = get_overlapping_files_percentage(tr_1, tr_2)
    val = get_overlapping_files_percentage(val_1, val_2)
    te = get_overlapping_files_percentage(te_1, te_2)

    svm = LinearSVC(verbose=1, max_iter=10000)
    svm.fit(tr, tr_y)
    acc = svm.score(val, val_y)




