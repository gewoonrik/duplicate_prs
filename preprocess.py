from multiprocessing import Pool
import re
from os import listdir
from os.path import isfile, join

def preprocess(str):
    # remove links
    str = re.sub(r'\[(.*?)\]\(.+?\)', "", str)
    # remove urls
    str = re.sub(r'http(s)?:\/\/\S*? ', "", str)
    # remove quotes
    str = re.sub(r'> (.*?)\n', "", str)
    # remove code
    str = re.sub(r'```[a-z]*?\n.*?\n```', "", str)
    # remove package names
    str = re.sub(r'[a-z][a-z0-9_]*(\.\S+)+[\S]', "", str)
    # remove paths
    str = re.sub(r'\/?\S+?\/\S+(\/\S*?)*?\/?', "", str)


    return str

path_from = "comments"
path_to = "comments_preprocessed"
files = [f for f in listdir(path_from) if isfile(join(path_from, f))]

def preprocess_file(file):
    f = open(path_from+"/"+file, 'r')
    str = f.read()
    f.close()

    str = preprocess(str)

    f = open(path_to+"/"+file, 'w')
    f.write(str)
    f.close()

p = Pool(16)
p.map(preprocess_file,files)
