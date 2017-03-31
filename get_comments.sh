mongo github get_pr_comments.js > pr_comments.csv
python filter_pr_references.py > pr_reference_comments.csv
python download_comments.py
python preprocess.py
