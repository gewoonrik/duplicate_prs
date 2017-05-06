from DuplicatePRs.dataset import get_diff_file
import os
import urllib
def download_diff(owner, repo, id):
    file = get_diff_file(owner, repo, id)

    if not os.path.isfile(file):
        url = "https://www.github.com/" + owner + "/" + repo + "/pull/" + str(id) + ".diff"
        urllib.urlretrieve(url, file)
