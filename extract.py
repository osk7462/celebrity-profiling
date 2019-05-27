import ndjson
import json
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import re



def preprocess(tweets):
	clean_tweets = ""
	tweet = ""
	for tweet in tweets:
		tweet = "".join(tweet)
		# print(tweet)
		tweet =  re.sub("<.*?>", " ", tweet) 
		tweet = re.sub("@.*? ", "", tweet)
		tweet = re.sub("#.*? ", "", tweet)
		tweet =  re.sub("<.*?>"," ", tweet) 
		tweet = re.sub("@.*?,","", tweet)
		tweet = re.sub("#.*?,","", tweet)
		tweet = re.sub("@.*?.","", tweet)
		tweet = re.sub("#.*?.", "", tweet)
		tweet = tweet.split(" ")
		if tweet[len(tweet)-1][:4] == "http":
			tweet = tweet[:len(tweet)-1]
			tweet = " ".join(tweet)
		else:
			tweet = " ".join(tweet)
			# print(tweet)
			tweet = re.sub("https.*? ", "", tweet)
			tweet = re.sub("https.*?.", "", tweet)
			tweet = re.sub("https.*?,", "", tweet)
		clean_tweets += tweet
	return clean_tweets


def create_feed_csv():		
	with open("feed.csv", "w") as csvfile:
		csv_writer = csv.DictWriter(csvfile, fieldnames = ["id", "tweet"])
		csv_writer.writeheader()
		with open('feeds.ndjson', "r") as f:
			for line in f:
				data = json.loads(line)
				csv_writer.writerow({"id": data["id"], "tweet": preprocess(data["text"][:20])})


def create_labels_csv():
	with open("labels.csv", "w") as labels:
		csv_writer = csv.DictWriter(labels, fieldnames=["id", "occupation", "gender", "fame",\
		 "birthyear"])
		csv_writer.writeheader()	
		with open("feed.csv", "r") as feed_csv:
			csv_reader = csv.reader(feed_csv)
			next(csv_reader)
			with open('labels.ndjson', "r") as f:
				data = ndjson.load(f)
				count = int(0)
				for cid in csv_reader:
					count += 1
					print(count)
					# print(cid[0])
					for i in range(len(data)):
						if str(data[i]["id"]) == str(cid[0]):
							data[i]["birthyear"] = str(2019-int(data[i]["birthyear"]))
							age = int(data[i]['birthyear'])
							if age > 10 and age < 20:
								age_class = 'A'
							elif age >=20 and age <35:
								age_class = 'B'
							elif age >=35 and age < 50:
								age_class = 'C'
							elif age >= 50 and age < 65:
								age_class = 'D'
							elif age >= 65 and age < 80:
								age_class = 'E'
							else:
								age_class = 'F'
							data[i]['birthyear'] = age_class
							csv_writer.writerow(data[i])
							data.pop(i)
							break

						
def NB(label):
	data_feeds = pd.read_csv("feed.csv")
	data_feeds = data_feeds.iloc[:, 1]
	data = pd.read_csv("labels.csv",) 
	if label == 'gender':
		y = data.iloc[:, 2:3]
	elif label == 'occupation':
		y = data.iloc[:, 1:2]
	elif label == 'fame':
		y = data.iloc[:, 3:4]
	elif label == 'birthyear':
		y = data.iloc[:, 4: ]
	X_train, X_test, y_train, y_test = train_test_split(data_feeds, y, test_size = 0.2, random_state = 0)
	# print(y_train)
	count_vect = CountVectorizer()
	X_train_counts = count_vect.fit_transform(X_train.values.astype('U'))
	X_train_counts.shape
	print(X_train_counts.shape)
	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
	X_train_tfidf.shape

	clf = MultinomialNB().fit(X_train_tfidf, y_train)

	X_test_counts = count_vect.transform(X_test.values.astype('U'))
	X_test_counts.shape
	X_test_tfidf = tfidf_transformer.transform(X_test_counts)
	X_test_tfidf.shape


	predicted = clf.predict(X_test_tfidf)
	accuracy = accuracy_score(y_test, predicted, normalize=True, sample_weight=None)
	print("\n\n{} accuracy: {}\n".format(label, accuracy))
	cm = confusion_matrix(y_test, predicted)
	print("\nclassification report\n {}\n\n".format(classification_report(y_test, predicted)))




if __name__ == '__main__':
	# create_feed_csv()
	# create_labels_csv()

	labels = ['gender', 'occupation', 'fame', 'birthyear']
	for label in labels:
		NB(label)