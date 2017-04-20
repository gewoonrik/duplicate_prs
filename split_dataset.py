import random

f = open("pairs.csv", "r")
lines = f.read().split("\n")
f.close()

lines = lines[1:len(lines)-1]
random.shuffle(lines)

count = len(lines)
split = int(0.1 * count)

#test validation training
# 0.1 0.1 0.8
test = lines[:split]
validation = lines[split: split+split]
training = lines[split+split:]

def write_lines(file, lines):
    f = open(file, "w")
    f.write("owner,repo,pr1_id,p2_id\n")
    for line in lines:
        f.write(line+"\n")
    f.close()

write_lines("test.csv", test)
write_lines("validation.csv", validation)
write_lines("training.csv", training)


