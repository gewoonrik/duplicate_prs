import urllib2

from DuplicatePRs.dataset import get_diff_file
import os
import urllib
def download_diff(owner, repo, id):
    file = get_diff_file(owner, repo, id)

    if not os.path.isfile(file):
        url = "https://www.github.com/" + owner + "/" + repo + "/pull/" + str(id) + ".diff"
        urllib.urlretrieve(url, file)
    return file
def download_diff_string(owner,repo,id):
    url = "https://www.github.com/" + owner + "/" + repo + "/pull/" + str(id) + ".diff"
    return urllib2.urlopen(url).read()