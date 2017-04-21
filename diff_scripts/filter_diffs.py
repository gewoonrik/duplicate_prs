#filter pairs that contain a diff that contain no diff (Sorry, this diff is unavailable. or zero lines)
# or is too large
from load_data import load_data, line_to_files

def is_valid_diff(file):
    f = open(file, "r")
    content = f.read()
    nr_lines = content.count("\n")
    f.close()
    return nr_lines > 0 and nr_lines <= 500

def filter_diffs_in_file(file):
    lines  = load_data(file)
    correct_lines = []
    for line in lines:
        pr1, pr2 = line_to_files(line)
        if is_valid_diff(pr1) and is_valid_diff(pr2):
            correct_lines.append(line)
    return correct_lines

def write_lines(file, lines):
    f = open(file, "w")
    f.write("\n".join(lines))
    f.close()

train_filtered = filter_diffs_in_file("training_with_negative_samples.csv")
write_lines("training_with_negative_samples_filtered.csv", train_filtered)

validation_filtered = filter_diffs_in_file("validation_with_negative_samples.csv")
write_lines("validation_with_negative_samples_filtered.csv", validation_filtered)

test_filtered = filter_diffs_in_file("test_with_negative_samples.csv")
write_lines("test_with_negative_samples_filtered.csv", test_filtered)
