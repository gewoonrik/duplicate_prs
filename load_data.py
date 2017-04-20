def load_data(file):
    f = open(file, "r")
    lines = f.read().split("\n")
    f.close()
    #remove head and empty line at bottom
    return lines[1:len(lines)-1]

def lines_to_files(lines):
    files = []
    for line in lines:
        owner, repo, pr1, pr2, is_dup = line.split(",")
        files.append("diffs/"+owner+"@"+repo+"@"+pr1+".diff")
        files.append("diffs/"+owner+"@"+repo+"@"+pr2+".diff")
    return files