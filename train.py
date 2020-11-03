# python3 script for to  extract data from the twitter based upon any product and do sentimental analysis.

import re 
import tweepy 
from tweepy import OAuthHandler 
from textblob import TextBlob 
import pandas as pd
import config
from sklearn.model_selection import train_test_split # function for splitting data to train and test sets
from nltk.corpus import stopwords
import numpy as np
import string
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
# ML Libraries
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# Global Parameters
stop_words = set(stopwords.words('english'))

class TwitterClient(object): 
    ''' 
    Generic Twitter Class for sentiment analysis. 
    '''
    def __init__(self): 
        ''' 
        Class constructor or initialization method. 
        '''
        self.config_twitter = config
        # attempt authentication 
        try: 
            # create OAuthHandler object 
            self.auth = OAuthHandler(self.config_twitter.consumer_key, self.config_twitter.consumer_secret) 
            # set access token and secret 
            self.auth.set_access_token(self.config_twitter.access_key, self.config_twitter.access_secret) 
            # create tweepy API object to fetch tweets 
            self.api = tweepy.API(self.auth) 
        except: 
            print("Error: Authentication Failed") 

    def clean_tweet(self, tweet): 
        ''' 
        Utility function to clean tweet text by removing links, special characters 
        using simple regex statements. 
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", tweet).split()) 

    def get_tweet_sentiment(self, tweet): 
        ''' 
        Utility function to classify sentiment of passed tweet 
        using textblob's sentiment method 
        '''
        # create TextBlob object of passed tweet text 
        clean_tweet = self.clean_tweet(tweet)
        analysis = TextBlob(clean_tweet)
        
        if analysis.sentiment.polarity > 0.5: 
            return 'positive', clean_tweet
        
        elif (analysis.sentiment.polarity <= 0.5 and analysis.sentiment.polarity > 0): 
            return 'slightly positive', clean_tweet
        
        elif analysis.sentiment.polarity == 0: 
            return 'neutral', clean_tweet
        
        elif analysis.sentiment.polarity < -0.5 and analysis.sentiment.polarity > 0: 
            return 'slightly negative', clean_tweet
        
        else: 
            return 'negative', clean_tweet

    def get_tweets(self, search_words,date_since, count): 
        ''' 
        Main function to fetch tweets and parse and store them in df. 
        Arguments :
            search_words : key word for searching tweet
            date_since : date from which tweets needed.
            count : number of tweets.
        
        Return :
            data : contains tweets and sentiments in the df format.
        
        '''
        c_tweet = []
        c_sentiment = []

        try: 
            # Calling api 
            api = tweepy.API(self.auth) 
    
            # Collect tweets
            fetched_tweets = tweepy.Cursor(api.search,
                        q=search_words,
                        lang="en",
                        since=date_since).items(count)

            # parsing tweets one by one 
            for tweet in fetched_tweets: 
                # empty dictionary to store required params of a tweet 
                parsed_tweet = {} 

                # saving text of tweet 
                parsed_tweet['text'] = tweet.text 
                # saving sentiment of tweet 
                parsed_tweet['sentiment'], parsed_tweet['clean_tweet'] = self.get_tweet_sentiment(tweet.text) 
                
                clean_tweet = parsed_tweet['clean_tweet']
                parsed_tweet['cleaned_tweet'] = self.preprocess_tweet_text(clean_tweet)
                
                # appending parsed tweet to tweets list 
                if tweet.retweet_count > 0: 
                    # if tweet has retweets, ensure that it is appended only once 
                    if parsed_tweet not in c_tweet: 
                        c_sentiment.append(parsed_tweet['sentiment'])
                        c_tweet.append(parsed_tweet['cleaned_tweet'])
                        
                else:
                    c_sentiment.append(parsed_tweet['sentiment'])
                    c_tweet.append(parsed_tweet['cleaned_tweet'])
                    
            # store data in data frame
            data = pd.DataFrame(
                        {'tweets': c_tweet, 'sentiments': c_sentiment})
            return data 

        except tweepy.TweepError as e: 
            # print error (if any) 
            print("Error : " + str(e)) 
            
            
    def preprocess_tweet_text(self, tweet):
        """ helper function to clean the tweets.
        """
        tweet = tweet.lower()
        # Remove urls
        tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
        # Remove user @ references and '#' from tweet
        tweet = re.sub(r'\@\w+|\#','', tweet)
        # Remove punctuations
        tweet = tweet.translate(str.maketrans('', '', string.punctuation))
        # Remove stopwords
        tweet_tokens = word_tokenize(tweet)
        filtered_words = [w for w in tweet_tokens if not w in stop_words]
        
        return " ".join(filtered_words)
    
    def get_feature_vector(self, train_fit):
        """ helper function which convert tokens to numbers
        """
        vector = TfidfVectorizer(sublinear_tf=True)
        vector.fit(train_fit)
        return vector
        
      
if __name__ == "__main__": 

    api = TwitterClient()
    # # get twitter data.
    get_data = api.get_tweets(config.search_words, config.date_since, config.count)
    print("get/-data", get_data)
    
    X, y = get_data['tweets'], get_data['sentiments']
    _, test_tweets, _, _ = train_test_split(X, y, test_size=0.2, random_state=30)
    
    tf_vector = api.get_feature_vector(np.array(X).ravel())
    X = tf_vector.transform(np.array(X).ravel())
    y = np.array(y).ravel()
    # training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

    # Training Naive Bayes model
    NB_model = MultinomialNB()
    NB_model.fit(X_train, y_train)
    y_predict_nb = NB_model.predict(X_test)
    NB_accuracy = accuracy_score(y_test, y_predict_nb)
    print("Accuracy for Naive Bayes: %.2f%%" % (NB_accuracy * 100.0))

    # Training Logistics Regression model
    LR_model = LogisticRegression(solver='lbfgs')
    LR_model.fit(X_train, y_train)
    y_predict_lr = LR_model.predict(X_test)
    LR_accuracy = accuracy_score(y_test, y_predict_lr)
    print("Accuracy for Logistics Regression: %.2f%%" % (LR_accuracy * 100.0))
    
    # Training  XGBoost model
    XGB_model = XGBClassifier()
    XGB_model.fit(X_train,y_train)

    # XgBoost prediction and scores
    y_pred_xgb = XGB_model.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
    print("Accuracy for XGBOOST: %.2f%%" % (xgb_accuracy * 100.0))
    
    # store's the results in csv
    testing_df = data = pd.DataFrame(
                        {'tweets': test_tweets, 'sentiments': y_pred_xgb})
    testing_df.to_csv("results.csv")
    
    
    
    
    
    
    
    
