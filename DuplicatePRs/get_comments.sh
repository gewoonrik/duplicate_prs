mkdir temp
mkdir comments
mkdir comments_preprocessed
mongo github comment_scripts/get_pr_comments.js > temp/pr_comments.csv
python comment_scripts/filter_pr_references.py > temp/pr_reference_comments.csv
python comment_scripts/download_comments.py
python comment_scripts/preprocess.py
