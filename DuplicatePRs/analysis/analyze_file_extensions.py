from os import listdir

from os.path import isfile, join

dir = "diffs"

files = [f for f in listdir(dir) if isfile(join(dir, f))]

def find_extensions(file):
    f = open(dir+"/"+file, "r")
    str = f.read()
    f.close()

    lines = str.split("\n")
    file_lines = [line for line in lines if line.startswith("--- a/")]

    return set([line.split("/")[-1].split(".")[-1] for line in file_lines])

counts = {}

file_extensions = [find_extensions(file) for file in files]

for exts in file_extensions:
    for ext in exts:
        if ext not in counts:
            counts[ext] = 1
        else:
            counts[ext] += 1

for value, key in sorted([(value,key) for (key,value) in counts.items()]):
    print key+": "+str(value)

