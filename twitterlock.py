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
        self.master_keywords = []
        self.master_feedback = {}

    def cycle1(self):
        self.master_keywords.extend(self.init_terms)
        fct.get_tweets(self.init_terms, self.size, self.filename)
        tweets_df = fct.process_tweets(self.filename)
        keywords = fct.semantic_indexing(tweets_df, self.master_feedback, self.size)
        fct.add_keywords_df(tweets_df, keywords)
        self.keywords = list(set(keywords))
        self.df = tweets_df


    def cycle2(self):
        #classify old dataframe based on new feedback
        fct.classify_tweets(self.df, self.feedback)
        columns = fct.keyword_binary_col(self.keywords, self.df)
        self.old_df = self.df

        fct.get_tweets(self.keywords + self.init_terms, self.size, self.filename)

        tweets_df = fct.process_tweets(self.filename)
        self.df = tweets_df

        #classify data from temporary new dataframe with model based on old dataframe
        fct.add_keywords_df(tweets_df, self.old_keywords)
        fct.keyword_binary_col(self.old_keywords, tweets_df)

        #commenting out testing

        #BEST_MODEL, BEST_PARAMS = fct.train_model_offline(self.old_df, columns)
        #fct.predict_classification(columns, self.old_df, tweets_df, BEST_MODEL, BEST_PARAMS)

        fct.predict_classification(columns, self.old_df, tweets_df)

        #prep for validation and next round
        self.df["classification"] = tweets_df["classification"]
        new_keywords = fct.semantic_indexing(tweets_df, self.master_feedback, self.size)
        fct.add_keywords_df(self.df, new_keywords)
        #final_keywords = fct.get_keywords(self.df)
        #self.keywords = list(set(final_keywords))
        self.keywords = new_keywords

    def take_feedback(self, feedback):
        self.feedback = feedback
        self.old_keywords = self.keywords
        self.keywords = fct.update_keywords(self.feedback)
        self.master_keywords.extend(self.keywords)
        self.master_feedback.update(self.feedback)

    def finish(self, filename):
        #final query and write tweets to filename
        good_words = fct.update_keywords(self.feedback)
        tweets = fct.get_tweets(good_words, self.size, filename)
        tweets_df, _ = fct.process_tweets(tweets)
        tweets_df.to_csv(filename)

    def set_satisfaction(self, response):
        self.satisfactory = response