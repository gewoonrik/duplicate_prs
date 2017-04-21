def load_data(file):
    f = open(file, "r")
    lines = f.read().split("\n")
    f.close()
    #remove head and empty line at bottom
    return lines[1:len(lines)-1]

def lines_to_files(lines):
    files = []
    for line in lines:
        pr1, pr2 = line_to_files(line)
        files.append(pr1)
        files.append(pr2)
    return files

def line_to_files(line):
    owner, repo, pr1, pr2, is_dup = line.split(",")
    pr1 = "diffs/"+owner+"@"+repo+"@"+pr1+".diff"
    pr2 = "diffs/"+owner+"@"+repo+"@"+pr2+".diff"
    return pr1, pr2