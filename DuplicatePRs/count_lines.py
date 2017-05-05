from os import walk

lines = []
for (dirpath, dirnames, filenames) in walk("diffs"):
    for file in filenames:
        f = open("diffs/"+file, "r")
        content = f.read()
        f.close()
        nrlines = content.count("\n")
        lines.append(nrlines)

f = open("nrlines.txt", "w")

txt = "\n".join(str(x) for x in lines)
f.write(txt)
f.close()