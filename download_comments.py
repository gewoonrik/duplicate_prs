import pymongo
from pymongo import MongoClient
import csv
import re
import sys

# only prints pr comments that reference another PR

client = MongoClient('127.0.0.1', 27017)
db = client.github

def progressBar(value, endvalue, bar_length=20):

        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        sys.stderr.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
        sys.stderr.flush()


with open('pr_reference_comments.csv', 'rb') as file:
        issues = file.read()
	lines = issues.split("\n")
	count = len(lines)
	i = 0
        for line in lines:
                owner, repo, issue_id, id = line.split(",")
                issue_comment = db.issue_comments.find_one({"repo":repo, "owner":owner, "issue_id":int(issue_id),"id":int(id)})
		f = open('comments/'+owner+'-'+repo+'-'+issue_id+'-'+id+'.txt', 'w')
		f.write(issue_comment["body"].encode('utf-8'))
		f.close()
		progressBar(i, count)
		i +=1
