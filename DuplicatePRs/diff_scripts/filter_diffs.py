#filter pairs that contain a diff that contain no diff (Sorry, this diff is unavailable. or zero lines)
# or is too large
from DuplicatePRs.dataset import load_csv, line_to_diff_files

def is_valid_diff(file):
    f = open(file, "r")
    content = f.read()
    nr_lines = content.count("\n")
    f.close()
    return nr_lines > 0 and nr_lines <= 28567

def filter_diffs_in_file(file):
    lines  = load_csv(file)
    correct_lines = []
    for line in lines:
        pr1, pr2, _ = line_to_diff_files(line)
        if is_valid_diff(pr1) and is_valid_diff(pr2):
            correct_lines.append(line)
    return correct_lines

def write_lines(file, lines):
    f = open(file, "w")
    f.write("\n".join(lines))
    f.close()

if __name__ == "__main__":
    train_filtered = filter_diffs_in_file("training_with_negative_samples.csv")
    write_lines("training_with_negative_samples_filtered.csv", train_filtered)

    validation_filtered = filter_diffs_in_file("validation_with_negative_samples.csv")
    write_lines("validation_with_negative_samples_filtered.csv", validation_filtered)

    test_filtered = filter_diffs_in_file("test_with_negative_samples.csv")
    write_lines("test_with_negative_samples_filtered.csv", test_filtered)
