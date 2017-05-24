from DuplicatePRs.dataset import get_diff_file, read_normal
import os
import urllib
def download_diff(owner, repo, id):
    file = get_diff_file(owner, repo, id)

    if not os.path.isfile(file):
        url = "https://www.github.com/" + owner + "/" + repo + "/pull/" + str(id) + ".diff"
        urllib.urlretrieve(url, file)
    return file
def download_diff_string(owner,repo,id):
    file = get_diff_file(owner, repo, id)
    if os.path.isfile(file):
        return read_normal(file)
    try:
        url = "https://www.github.com/" + owner + "/" + repo + "/pull/" + str(id) + ".diff"
        content = urllib.urlopen(url).read()
        f = open("file", "w")
        f.write(content)
        f.close()
        return content
    except:
        return ""