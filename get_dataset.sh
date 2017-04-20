echo "generating dataset"
python generate_dataset.py
cat pairs.csv | sort -u > pairs2.csv
rm pairs.csv
mv pairs2.csv pairs.csv
echo "getting diffs"
python download_diffs.py
echo "removing pairs without diffs"
python check_diffs.py
python split_dataset.py
