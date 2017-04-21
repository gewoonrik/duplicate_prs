mkdir diffs
echo "generating dataset"
python diff_scripts/generate_dataset.py
cat temp/pairs.csv | sort -u > temp/pairs2.csv
rm temp/pairs.csv
mv temp/pairs2.csv temp/pairs.csv
echo "getting diffs"
python diff_scripts/download_diffs.py
echo "removing pairs without diffs"
python diff_scripts/check_diffs.py
python diff_scripts/split_dataset.py
python diff_scripts/generate_negative_samples.py