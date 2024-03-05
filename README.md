Run the code:
python Sentiment_Analysis_Video_Games.py – trains the machine learning model
python Twitter_Sentiment_Analysis.py – applies the machine learning model and calculates the sentiments of the tweets
Make sure the dataset.csv and Star_Wars_Tweets.csv is in the same directory while running the code
List of required libraries:
Pandas
Random
NLTK 
-	import nltk
-	from nltk.corpus import stopwords
-	from nltk.stem import WordNetLemmatizer
-	nltk.tokenisze import word_tokenize
Textblob
Sklearn
-	from sklearn.model_selection import train_test_split
-	from sklearn.feature_extraction.text import TfidfVectorizer
-	from sklearn.svm import LienarSVC
-	from sklearn.model_selection import LogisticRegression
Pickle
Word Cloud
MatplotLib
Re
Contractions
From googletrans import Translator
Emoji


