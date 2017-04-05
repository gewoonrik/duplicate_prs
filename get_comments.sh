mongo github get_pr_comments.js > temp/pr_comments.csv
python filter_pr_references.py > temp/pr_reference_comments.csv
python download_comments.py
python preprocess.py
