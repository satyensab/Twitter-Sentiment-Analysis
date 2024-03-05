# Import libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import seaborn as sns
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore")


def cal_statistics(df):
    # In our data set alot of the review our based on reviews from Terraria, PAYDAY 2, DayZ, Dota 2, etc.
    distinct = df['app_name'].value_counts(sort=True)
    ax = distinct[:30].plot(kind="bar", figsize=(15, 6), rot=45)
    ax.set_xlabel('Game Name')
    ax.set_ylabel('Count')
    ax.set_title('Game Count')
    # Show the plot
    plt.show()

    # There are more positive game review than negative game reviews. We have to keep this in mind when doing our
    # analysis
    df['review_score'].value_counts().plot(kind='bar')


def sentiment_analysis(df):
    # Now use nltk and textblob to calculate sentiment and then compare it to the actual sentiment
    nltk.download(
        'stopwords')  # downloads all the stop words from nltk, words that are not useful when calculating sentiment
    nltk.download('wordnet')
    stop_words = stopwords.words('english')
    # print(stop_words)


    good_rating = ['6', '7', '8', '9', '10']
    bad_rating = ['0', '1', '2', '3', '4']
    punctuation = ['!', '.', '$', '#', '@', '/', ';', ':', '^', ')', '(', '&']
    lemmatize = WordNetLemmatizer()
    cleaned_text = []
    for review in df['review_text']:
        cleaned_review = ""
        for word in word_tokenize(review):
            word = word.lower()
            word = lemmatize.lemmatize(word)
            if word in good_rating:
                word = 'great'
            if word in bad_rating:
                word = 'horrible'
            if word not in stop_words and word not in punctuation:
                cleaned_review += word + " "
        cleaned_text.append(cleaned_review)
    df['cleaned_review_text'] = cleaned_text

    df['sentiment'] = df['review_text'].apply(lambda review: TextBlob(review).sentiment.polarity)
    df['pred_review_score'] = df['sentiment'].apply(lambda x: 1 if x >= 0.00 else -1)

    Accuracy_Score = accuracy_score(df['review_score'], df['pred_review_score'])
    print("Accuracy_Score NLTK: " + str(round(Accuracy_Score, 4) * 100) + "%")
    # Accuracy_Score : 76.1%
    # Accuracy improved to 76.1% when more in depth pre-processing was done


def plot_plots(df):
    # Visualizing real review score to what the text blob sentiment outputted
    sns.set_theme()
    sns.set_style('darkgrid')
    df['Sentiment'] = df['pred_review_score'].apply(lambda x: 'Positive' if x == 1 else 'Negative')
    df['subjective'] = df['review_text'].apply(lambda review: TextBlob(review).sentiment[1])

    sns.relplot(data=df, x='sentiment', y='subjective', hue='Sentiment', legend=True).set(
        title="Sentiment vs Subjectivity of Steam Game Reviews")
    plt.show()
    plt.close()
    # Add a word cloud for both the positive and negative reviews, to see which words were used most frequently for each
    pos_review_word = df[
        df['pred_review_score'] == 1]  # (Not used in analysis as we are analyzing the tweeter data not the steam data)
    positive_reviews = pos_review_word['cleaned_review_text'].to_string(index=False)
    #
    neg_review_word = df[df['pred_review_score'] == -1]
    negative_reviews = neg_review_word['cleaned_review_text'].to_string(index=False)

    # Positive Word Cloud (used in twitter analysis)
    positive_word_cloud = WordCloud(width=350, height=350, background_color='black', max_words=50,
                                    min_font_size=10).generate(positive_reviews)
    #
    plt.imshow(positive_word_cloud)
    plt.axis("off")
    plt.show()

    # Negative Word Cloud
    negative_word_cloud = WordCloud(width=350, height=350, background_color='black', max_words=50,
                                    min_font_size=10).generate(negative_reviews)
    plt.imshow(negative_word_cloud)
    plt.axis("off")
    plt.show()


def create_sentiment_model(df):
    # Using scilearn to predict sentiment, testing out the features parameter to give the best accuracy
    num_features = [500, 1000, 2000, 3000, 4000, 5000, 7000, 10000]
    accuracies = []
    for feature in num_features:
        tf = TfidfVectorizer(max_features=feature)

        X = tf.fit_transform(df['cleaned_review_text'])
        x_train, x_test, y_train, y_test = train_test_split(X, df['review_score'], test_size=0.25, random_state=31)
        LinearSVC_ml = LinearSVC(random_state=31)
        LinearSVC_ml.fit(x_train, y_train)
        pred = LinearSVC_ml.predict(x_test)

        Accuracy_Score = accuracy_score(y_test, pred)
        accuracies.append(Accuracy_Score)
    print(accuracies)

    # Plot the Features vs Accuracy (which gives the best accuracy will be our model)
    features_plot = sns.lineplot(x=num_features, y=accuracies)
    features_plot.set(xlabel="Number of Features", ylabel="Accuracy", title='Number of Features vs Accuracy')

    # Instead of manually testing different hyper parameters such as the number of features we did above we can apply
    # GridSearchCV which exhaustively considers all parameter combinations to give us the best model
    tf = TfidfVectorizer(max_features=2000)

    X = tf.fit_transform(df['cleaned_review_text'])
    x_train, x_test, y_train, y_test = train_test_split(X, df['review_score'], test_size=0.25, random_state=31)

    param_grid = [
        {'C': [0.1, 0.3, 0.5, 0.7, 1, 10, 100], 'max_iter': [100, 200, 500, 1000]}
    ]

    model = LinearSVC(random_state=31)
    Best_Model = GridSearchCV(estimator=model, param_grid=param_grid)
    Best_Model.fit(x_train, y_train)
    pred_Y = Best_Model.predict(x_test)

    print("Accuracy_Score SKLEARN(Linear SVC): " + str(Best_Model.best_score_))
    print("Best Estimators: " + str(Best_Model.best_estimator_))
    # Confusion Matrix
    print(confusion_matrix(y_test, pred_Y))
    print(classification_report(y_test, pred_Y))

    cm = confusion_matrix(y_test, pred_Y)
    matrix = ConfusionMatrixDisplay(confusion_matrix=cm)
    matrix.plot()

    X = tf.fit_transform(df['cleaned_review_text'])
    x_train, x_test, y_train, y_test = train_test_split(X, df['review_score'], test_size=0.2, random_state=31)
    LogisticRegression_MD = LogisticRegression(random_state=31)
    Logistic_Regression_Best = GridSearchCV(estimator=LogisticRegression_MD, param_grid=param_grid)
    Logistic_Regression_Best.fit(x_train, y_train)
    pred_Y_Logistic = Logistic_Regression_Best.predict(x_test)

    print("Accuracy_Score SKLEARN(Logistic Regression): " + str(Logistic_Regression_Best.best_score_))
    print("Best Estimators: " + str(Logistic_Regression_Best.best_estimator_))

    # Save Scit-Learn model to use for our twitter sentiment analysis
    filename = "Sentiment_Model_LinearSVC.sav"
    filename1 = "Vectorizer.sav"
    pickle.dump(Best_Model, open(filename, 'wb'))
    pickle.dump(tf, open(filename1, 'wb'))


if __name__ == "__main__":
    # Read In Data
    # Reads in dataset
    random.seed(10)
    subset_range = sorted(
        random.sample(range(1, 6417106),
                      6417106 - 50000))  # will be using 50,000 data points for our sample for smaller loading times
    df = pd.read_csv("dataset.csv", header=0, skiprows=subset_range)

    # Reads in and drops any rows with NA values
    df = df.dropna()

    print("\nCalculating statistics.....")
    cal_statistics(df)
    print("\nPerforming sentiment analysis on data using TextBlob....")
    sentiment_analysis(df)
    print("\nCreating plots...")
    plot_plots(df)
    print("Creating custom sentiment model and calculating accuracy....")
    create_sentiment_model(df)
