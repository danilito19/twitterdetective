import functions as fct
import pandas as pd

class Twitterlock:
    def __init__(self, words=None, size=20, filename="temp.txt"):
        self.init_terms = words
        self.keywords = None
        self.old_keywords = None
        self.tweets = None
        self.df = None
        self.old_df = None
        self.size = size
        self.satisfactory = False
        self.feedback = None
        self.filename = filename

    def cycle1(self):
        fct.get_tweets(self.init_terms, self.size, self.filename)
        tweets_df = fct.process_tweets(self.filename)
        keywords = fct.semantic_indexing(tweets_df, self.size)
        fct.add_keywords_df(tweets_df, keywords)
        self.keywords = list(set(keywords))
        self.df = tweets_df


    def cycle2(self):
        #classify old dataframe based on new feedback
        fct.classify_tweets(self.df, self.feedback)
        columns = fct.keyword_binary_col(self.keywords, self.df)
        print("binary 1 passed")
        self.old_df = self.df

        #get new dataframe and set to self.df
        ''' Keywords is empty at this point'''

        fct.get_tweets(self.keywords, self.size, self.filename)

        tweets_df = fct.process_tweets(self.filename)
        print("tweets_df processed")
        print(tweets_df.head)
        self.df = tweets_df

        #classify data from temporary new dataframe with model based on old dataframe
        fct.add_keywords_df(tweets_df, self.old_keywords)
        fct.keyword_binary_col(self.old_keywords, tweets_df)
        print("binary 2 passed")
        print(self.old_df)
        self.old_df.to_csv("old_df.csv", sep=',', encoding='utf-8')
        fct.train_model_offline(self.old_df, columns)

        #fct.predict_classification(columns, self.old_df, tweets_df)

        #prep for validation and next round
        self.df["classification"] = tweets_df["classification"]
        new_keywords = fct.semantic_indexing(tweets_df, self.size)
        fct.add_keywords_df(self.df, new_keywords)
        final_keywords = get_keywords(self.df)
        self.keywords = list(set(final_keywords))
        self.tweets = tweets

    def take_feedback(self, feedback):
        self.feedback = feedback
        self.old_keywords = self.keywords
        self.keywords = fct.update_keywords(self.feedback)

    def finish(self, filename):
        #final query and write tweets to filename
        good_words = fct.update_keywords(self.feedback)
        query = fct.build_query(good_words)
        tweets = fct.get_tweets(query, self.size)
        tweets_df, _ = fct.process_tweets(tweets)
        tweets_df.to_csv(filename)

    def set_satisfaction(self, response):
        self.satisfactory = response