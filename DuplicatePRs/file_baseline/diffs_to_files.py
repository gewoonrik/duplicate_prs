from multiprocessing import Pool

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
    pr_lines = filter_file_lines(string)
    return map(file_line_to_file, pr_lines)

def get_overlapping_file_percentage(pr1, pr2):
    pr1 = file_to_files(pr1)
    pr2 = file_to_files(pr2)
    intersection = len([x for x in pr1 if x in pr2])
    total = (len(pr1)+len(pr2))*1.0
    if total == 0:
        return 1.0
    return intersection/total

def line_to_overlapping_file_percentage(line):
    pr1, pr2, is_dup = line_to_diff_files(line)
    return get_overlapping_file_percentage(pr1,pr2), is_dup

if __name__ == "__main__":

    p = Pool(32)

    tr, tr_y = zip(*p.map(line_to_overlapping_file_percentage, load_csv(config.training_dataset_file)))
    val, val_y = zip(*p.map(line_to_overlapping_file_percentage,load_csv(config.validation_dataset_file)))
    te, te_y = zip(*p.map(line_to_overlapping_file_percentage, load_csv(config.test_dataset_file)))


    svm = LinearSVC(verbose=1, max_iter=10000)
    svm.fit(tr, tr_y)
    acc = svm.score(val, val_y)




