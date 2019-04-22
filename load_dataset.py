import sqlite3
import csv
import json

conn = sqlite3.connect('./db.sqlite3')
c = conn.cursor()

links = []
links_path='./dataset/ml-20m/ml-20m/links.csv'
movies_path='./dataset/ml-20m/ml-20m/movies.csv'
tags_path='./dataset/ml-20m/ml-20m/tags.csv'
ratings_path='./dataset/ml-20m/ml-20m/ratings.csv'


with open(tags_path, 'r', encoding="utf8") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    name = None
    for row in reader:
        if row[0]!='userId':
            #genres =  json.dumps(row[2].split('|'))
            print(row)
            c.execute("INSERT INTO movies_tag(user_id, movie_id, tag, timestamp) VALUES(?, ?, ?, ?)", (row[0], row[1], row[2], row[3]))


conn.commit()
conn.close()
