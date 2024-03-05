# Import Libraries
import tweepy as tw
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import seaborn as sns
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pickle
import re
import contractions
from googletrans import Translator
import emoji


# retrieve Twitter API Key (because their is a query limit, I have already provided the tweeter data in the directory, this is just how I did it
# def retrieve_tweets():
#     consumer_key = 'ed3zJTxlTk8ocVAqgg3NtBfTe'
#     bearer_token = 'AAAAAAAAAAAAAAAAAAAAAPn9mwEAAAAAo1q64%2FDK3z%2Br2vpgI4pbNCvVLFU%3D2YV70psvQTFHAZfMiDGOBbgvdl3j28YT39lAygMi7V6hKQ8udQ'
#     consumer_secret = 'tM0gzjaJbzworjY8PPYGoCJnNMdkNV5w6UgxWxDbJZixuzmhpg'
#     access_token = '1651005753153519616-ck9SDaj71TV0mWXH88StTvlgcOAHHT'
#     access_token_secret = 'yLldhCiM55YzrTPTc6nMFnjijBdEJzxYjIz8y70d0WiNW'
#
#     #Authenticate to Twitter
#     auth = tw.OAuthHandler(consumer_key, consumer_secret)
#     auth.set_access_token(access_token, access_token_secret)
#     api = tw.API(auth, wait_on_rate_limit=True)
#
#     #Get tweets and store them into a dictionary
#     hashtag = '#StarWarsJediSurvivor'
#     tweets = tw.Cursor(api.search_tweets, q=hashtag).items(1000)
#     tweets = [{'Tweets': tweet.text, 'Timestamp': tweet.created_at} for tweet in tweets]
#     #Code for using Twitter API,authenticating to Twitter, and putting it into a dictionary is from
#     #Resource: https://www.youtube.com/watch?v=_EgqxIoUE7U&list=LL&index=6&t=489s&ab_channel=NicholasRenotte
#
#     #Converts the created dictionary to dataframe and csv for use
#     df = pd.DataFrame.from_dict(tweets)
#     df.to_csv("Star_Wars_Tweets.csv", index=False)
#
#     #Code Above is how I was able to accces tweets using the twitter api, as there is a limit query and quering it take
#     #some time I converted it into a dataframe. But this csv I created is not from a source online but a dictionary I
#     #created using the tutorial mentioned above

def pre_processing(df):
    # Gets stopwords
    nltk.download('stopwords')
    nltk.download('wordnet')
    stop_words = stopwords.words('english')

    # Preprocessing/Cleaning Text before applying sentiment analysis
    custom_stop_words = ['?', ':', ',', 'red_circle', 'camera_with_flash', 'wrapped_gift', 'ringed_planet', 'live',
                         'star',
                         'war', 'contest', 'jedi', 'survivor', 'pushpin', 'giveaway', 'crescent_moon', 'stream',
                         'game', 'hey']  # words that dont provide any meaning to the text
    good_sentiment = ['glowing_star', 'winbeaming_face_with_smiling_eyes', 'musical_noteslet', 'ok_hand',
                      'partying_facefire', 'firewrapped_gift', 'smiling_face_with_heart-eyes', 'thumbs-up', 'red-heart',
                      'purple-heart', 'firecrossed_swords', 'speaker-high-volume', 'keycap_1', 'face-with_symbols_on',
                      'backhand_index_pointing_right', 'pinched-fingers', '10',
                      'grinning_squinting_face']  # words that were only identified as positive after analyzing the
    # dataset
    lemmatize = WordNetLemmatizer()
    translator = Translator()

    cleaned_array = []
    for tweet in df['Tweets']:
        cleaned_tweet = ""
        tweet = translator.translate(tweet).text  # Translate text to english
        tweet = re.sub(r'#\w+|http\S+|rt|@\w+|', '', tweet,
                       flags=re.IGNORECASE)  # uses regular expressions to remove any word that starts with @, http, #,
        # and rt (also common symbols that are unnecessary for tweets)
        tweet = contractions.fix(tweet)  # fixes words like It's to it is
        tweet = emoji.demojize(tweet, delimiters=("", ""))  # converts emojis to text
        for word in word_tokenize(tweet):
            word = word.lower()
            if word in good_sentiment:
                word = 'great'
            if word not in custom_stop_words and word not in stop_words:
                cleaned_tweet += lemmatize.lemmatize(word) + " "
        cleaned_array.append(cleaned_tweet)

    df['Cleaned_Tweets'] = cleaned_array

    # Now that we have added some preprocessing tweets that only contained essentially promotion are now empty when
    # cleaned and should be deleted
    df['Cleaned_Tweets'].replace('', np.nan, inplace=True)
    df.dropna(subset=['Cleaned_Tweets'], inplace=True)


def calculate_sentiment(df):
    # Calculates the sentiment and polarity of the processed/clean tweets
    df['sentiment'] = df['Cleaned_Tweets'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
    df['subjectivity'] = df['Cleaned_Tweets'].apply(lambda tweet: TextBlob(tweet).sentiment[1])

    # Compute some aggregate statistics for the sentiments using NLTK and TextBlob
    print("\nStatistics")
    print("Sentiment")
    print(df['sentiment'].describe())
    print("Subjectivity")
    print(df['subjectivity'].describe())

    print("Predicted Sentiments")
    df['Label'] = df['sentiment'].apply(lambda x: "Positive" if x > 0 else "Negative" if x < 0 else "Neutral")
    distribution = df['Label'].value_counts()
    print(distribution)


def create_plots(df):
    # Plot Distribution of Positive, Negative, and Nuetral Tweets
    df['Label'].value_counts().plot(kind='pie')

    # Plots Sentiment vs Subjectivity
    sns.relplot(data=df, x='sentiment', y='subjectivity', hue=df['Label'], legend=True).set(
        title="Sentiment vs Subjectivity")
    plt.show()
    plt.close()
    # Show word cloud of what words were most used in
    pos_review_word = df[df['Label'] == "Positive"]

    positive_reviews = pos_review_word['Cleaned_Tweets'].to_string(index=False)

    positive_word_cloud = WordCloud(width=350, height=350, background_color='black', max_words=50,
                                    min_font_size=10).generate(positive_reviews)
    plt.imshow(positive_word_cloud)
    plt.axis("off")
    plt.show()

    # Words that were used in nuetral statements
    nue_review_word = df[df['Label'] == "Neutral"]
    nuetral_reviews = nue_review_word['Cleaned_Tweets'].to_string(index=False)

    nuetral_word_cloud = WordCloud(width=350, height=350, background_color='black', max_words=50,
                                   min_font_size=10).generate(nuetral_reviews)
    plt.imshow(nuetral_word_cloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
    # Words that were most used in negative tweets about the game
    negative_review_word = df[df['Label'] == "Negative"]
    negative_reviews = negative_review_word['Cleaned_Tweets'].to_string(index=False)

    negative_word_cloud = WordCloud(width=350, height=350, background_color='black', max_words=50,
                                    min_font_size=10).generate(negative_reviews)
    plt.imshow(negative_word_cloud)
    plt.axis("off")
    plt.show()

    # Look at the distribution of how well people liked/disliked it
    labels = ['Very Positive', 'Positive', 'Mildy Positve', 'Nuetral', 'Midly Negative', 'Negative', 'Very Negative']
    all_tweets = len(df['sentiment'])
    very_positive = len(df[df['sentiment'] >= 0.8]) / all_tweets
    positive = len(df[(df['sentiment'] >= 0.5) & (df['sentiment'] < 0.8)]) / all_tweets
    kinda_positive = len(df[(df['sentiment'] >= 0.0) & (df['sentiment'] < 0.5)]) / all_tweets
    neutral = len(df[df['sentiment'] == 0]) / all_tweets
    kinda_negative = len(df[(df['sentiment'] < 0.0) & (df['sentiment'] >= -0.5)]) / all_tweets
    negative = len(df[(df['sentiment'] <= -0.5) & (df['sentiment'] > -0.8)]) / all_tweets
    very_negative = len(df[df['sentiment'] <= -0.8]) / all_tweets

    Attitude = [very_positive, positive, kinda_positive, neutral, kinda_negative, negative, very_negative]

    sns.barplot(x=labels, y=Attitude, hue=labels).set(title="Distribution of Tweets")
    plt.xticks(rotation=45)
    plt.show()
    plt.close()
    # I conclude that the sentiment on the game is pretty good. Based on the word cloud some negatives seem to be the
    # performance, something negative to do with the pc port of the game, and it is to long. Some positives of the game
    # seem to be great exploration and how the game looks

    # Another plot to show the sentiment of release day
    ax = sns.lineplot(data=df, x="Timestamp", y="sentiment")
    ax.set_xticks([])
    plt.show()

    # Look at the distribution of the negative and positive tweets
    df_histo = df[df["sentiment"] != 0.000000]
    sns.histplot(data=df_histo, x="sentiment").set(title="Distribution of sentiment")
    plt.show()
    plt.close()

    df_histo_sub = df[df["subjectivity"] != 0.000000]
    sns.histplot(data=df_histo_sub, x="subjectivity").set(title="Distribution of subjectivity")
    plt.show()
    plt.close()

    positive_sentiment_hist = df[df['sentiment'] >= 0.01]
    sns.histplot(data=positive_sentiment_hist, x="sentiment").set(title="Distribution of positive-sentiment")
    plt.show()
    plt.close()

    negative_sentiment_hist = df[df['sentiment'] <= -0.01]
    sns.histplot(data=negative_sentiment_hist, x="sentiment").set(title="Distribution of negative-sentiment")
    plt.show()
    plt.close()

    sns.boxplot(data=df, x="Label", y="subjectivity").set(title="How meaningfull are the positive and negative tweets?")
    plt.show()
    plt.close()

    # Looking at different categories that you look for in a game and what are the sentiments around that (Ex: Is
    # Performance seen has more positive, neutral, or negative)
    df_Performance = df[
        df['Tweets'].str.contains('PERFORMANCE', regex=False, case=False) | df['Tweets'].str.contains('PERFORMING',
                                                                                                      regex=False,
                                                                                                      case=False)]
    df_Performance['Label'].value_counts()

    df_Quality = df[
        df['Tweets'].str.contains('QUALITY', regex=False, case=False) | df['Tweets'].str.contains('LOOK', regex=False,
                                                                                                  case=False)]
    df_Quality['Label'].value_counts()
    df_Gameplay = df[df['Tweets'].str.contains('GAMEPLAY', regex=False, case=False)]
    df_Gameplay['Label'].value_counts()
    df_Story = df[df['Tweets'].str.contains('JOURNEY', regex=False, case=False)]
    df_Story['Label'].value_counts()

    categories_plot = pd.DataFrame({
        "Positive": [3, 26, 3, 6],
        "Negative": [2, 1, 0, 1],
        "Neutral": [11, 4, 11, 4]},
        index=["Performance", "Quality", "Gameplay", "Journey/Story"])

    colors = ["green", "red", "gray"]
    categories_plot.plot(kind='bar', stacked=True, color=colors)
    plt.title("Sentiment Through Different Categories")
    plt.xlabel("Categories")
    plt.show()
    plt.close()


# Lets see how our custom model compares
def loading_sentiment_model(df):
    # loads the saved model
    LinearSVC_Model = pickle.load(open('Sentiment_Model_LinearSVC.sav', 'rb'))
    tf_Vectorizer = pickle.load(open('Vectorizer.sav', 'rb'))

    # Predicts on the model
    tweet_fit = tf_Vectorizer.transform(df['Cleaned_Tweets'])  # Transform documents to document-term matrix.
    df['sentiment_SVC'] = LinearSVC_Model.predict(tweet_fit)

    df['sentiment_SVC'] = df['sentiment_SVC'].apply(lambda x: "Positive" if x == 1 else "Negative")
    print(df['sentiment_SVC'].value_counts())
    df['sentiment_SVC'].value_counts().plot(kind='pie')
    plt.show()
    plt.close()
    print("Predicted Sentiments")

    print("Generating Word Clouds....")
    # Positive Word Cloud for LinearSVC Model
    pos_review_word_MD = df[df['sentiment_SVC'] == "Positive"]

    positive_reviews_MD = pos_review_word_MD['Cleaned_Tweets'].to_string(index=False)

    positive_word_cloud_MD = WordCloud(width=350, height=350, background_color='black', max_words=50, min_font_size=10,
                                       contour_width=3).generate(positive_reviews_MD)
    plt.imshow(positive_word_cloud_MD)
    plt.axis("off")
    plt.show()

    # Negative Word Cloud for LinearSVC Model
    neg_review_word_MD = df[df['sentiment_SVC'] == "Negative"]

    negative_reviews_MD = neg_review_word_MD['Cleaned_Tweets'].to_string(index=False)

    negative_word_cloud_MD = WordCloud(width=350, height=350, background_color='black', max_words=50,
                                       min_font_size=10).generate(negative_reviews_MD)
    plt.imshow(negative_word_cloud_MD)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    # Read In Data
    df = pd.read_csv("Star_Wars_Tweets.csv")
    print("\nPreprocessing Tweets....")
    pre_processing(df)
    print("\nCalculating Sentiment....")
    calculate_sentiment(df)
    print("\nGenerating Plots...")
    create_plots(df)
    print("\nLoading custom sentiment model...")
    loading_sentiment_model(df)
