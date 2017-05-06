from itertools import chain, imap
# counts the number of lines of the PRs in the duplicate pairs
from DuplicatePRs.dataset import load_csv, _current_path, get_diff_file

training = load_csv(_current_path+"/training.csv")
validation = load_csv(_current_path+"/validation.csv")
test = load_csv(_current_path+"/test.csv")
all = training + validation + test

def flatmap(f, items):
    return chain.from_iterable(imap(f, items))

def get_files_by_line(line):
    owner, repo, pr1, pr2 = line.split(",")
    return [get_diff_file(owner,repo,pr1), get_diff_file(owner, repo, pr2)]

files = flatmap(get_files_by_line, all)

lines = []
for file in files:
    f = open(file, "r")
    content = f.read()
    f.close()
    nrlines = content.count("\n")
    lines.append(nrlines)

f = open(_current_path+"/analysis/nrlines.txt", "w")

txt = "\n".join(str(x) for x in lines)
f.write(txt)
f.close()