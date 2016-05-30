from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, ensemble
from sklearn.metrics import *
from sklearn.cross_validation import train_test_split, KFold
from autorizador import *

import json
import string
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re


clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=0),
    'LR': LogisticRegression(random_state=0, n_jobs=-1),
    'SVM': svm.LinearSVC(random_state=0, dual= False),
    'NB': GaussianNB(),
    'KNN': KNeighborsClassifier(n_jobs = -1),
        }

grid = { 
'RF':{'n_estimators': [1,10,100], 'max_depth': [1,5,10,20,50,75], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,5]},
'NB' : {},
'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1], 'penalty': ['l1', 'l2']},
'KNN' :{'n_neighbors': [1, 3, 5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
       }


def get_credents():
	creds = get_creds('secrets.txt')
	auth=twitter.OAuth(creds.AccessToken,creds.AccessTokenSecret,creds.ConsumerKey, creds.ConsumerKeySecret)
	return auth

def save_tweet(tweet, f):
    # Replace HTML entities; function extracted from Borja Sotomayor Twitter Harvester 
    tweet['text'] = tweet['text'].replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
    json.dump(tweet, f)
    f.write('\n')



def save_user_tweets(user, n, auth):
    t = twitter.Twitter(auth=auth)
    print("Fetching %i tweets from @%s" % (n, user))
    tweets = t.statuses.user_timeline(screen_name=user, count=n)
    print("  (actually fetched %i)" % len(tweets))
    for tweet in tweets:
        save_tweet(tweet, outfile)
        print(tweet['text'])


# code to test that save is working
#users = ['aurelionuno']
#outfile = open('data.json', 'w')
# save_user_tweets('aurelionuno', 10, auth)


# read the filters file to track
# infile = open('filters_file.txt', 'r')
# track = ",".join(infile.read().strip().split("\n"))

def build_query(num_tweets, auth, filters_file=None, filter_words=None):
    '''
	Person Responsible: Manu Aragones

	+ filter_file: list of words / phrases to be included in query IN FILE
	+ filter_words: if filter_file not specified -> you can input filter manually
	+ num_tweets: the maximum number of tweets before break
	builds query for twitter API from user input

    Simplified fuction from Borja Sotomayor Twitter Harvester

    AND SAVES TO JSON_OUT OR RETURNS TO BE USED BY OTHER FUNCTIONS ?
	'''

    outf = open('json_out', "w")
    # Connect to the stream
    twitter_stream = twitter.TwitterStream(auth=auth)
    if filter_words is None and filters_file is None:
        stream = twitter_stream.statuses.sample()
    else:
        if filter_words is not None:
            track = filter_words
        elif filters_file is not None:
            infile = open(filters_file, 'r')
            track = ",".join(infile.read().strip().split("\n"))
        stream = twitter_stream.statuses.filter(track=track)
    # Fetch the tweets
    fetched = 0

    if num_tweets > 0:
        if outf != sys.stdout: print("Fetching %i tweets... " % num_tweets)
    else:
        signal.signal(signal.SIGINT, signal_handler)
        now = datetime.now().isoformat(sep=" ")
        msg = "[{}] Fetching tweets. Press Ctrl+C to stop.".format(now)
        if outf != sys.stdout: print(msg)

    for tweet in stream:
        print(tweet['text'])
        # The public stream includes tweets, but also other messages, such
        # as deletion notices. We are only interested in the tweets.
        # See: https://dev.twitter.com/streaming/overview/messages-types
        if tweet.get('text'):
            # We also only want English tweets
            if tweet['lang'] == "es":
                save_tweet(tweet, outf)
                fetched += 1
                if fetched % num_tweets == 0:
                    now = datetime.now().isoformat(sep=" ")
                    msg = "[{}] Fetched {:,} tweets.".format(now, fetched)
                    if outf != sys.stdout: print(msg)
                if num_tweets > 0 and fetched >= num_tweets:
                    break


def get_tweets(query, size, to_file=False):
    '''
    Person Responsible: Manu Aragones

    + query: string, query for twitter API
    + size: number of tweets desired
    + other arguments?

    Takes query, queries twitter API, returns JSON of tweets
    '''

    if not to_file:
        # get this part to return the tweets, not in file
        # and to take words from user interfase in main.py
        build_query(2, auth, 'words from main.py')
        #return tweets_raw

    else:
        # save to output file
        # this function dumps the tweets to json file in folder
        build_query(2, auth)

def cycle1(word_list):
    for word in word_list:
        print(word)
    return ["alpha", "beta", "theta"], None

def cycle2(feedback_dict, tweet_df):
    return ["crimson", "calico", "qwark"], None


def process_tweets(tweets_raw, tweets_random = None):
    '''
    Person Responsible: Devin Munger

    + tweets_raw: JSON of tweets as they are returned from API based on query
    + tweets_random: JSON of random sample of tweets as they are returned from API
    
    - If this is phase 1:
    tweets_random should be provided. tweets_raw and tweets_random should be added
    to one dataframe. Classification should be added. Tweets from tweets_random
    should be classified as not relevant, tweets from tweets_raw should be classified
    as relevant

    - If this phase 2:
    Tweets random should be translated into a dataframe. No classification added.

    - In either phase:
    Extract text from tweets_raw, write to file or create single string / dataframe
    (Do we want tweets to be kept distinct?)
    (Do we need to also output text of not relevant tweets for elimination purposes?)
    Output forrmat TBD by semantic processing person

    NEED TWEETS TO BE CONVERTED TO WORD COUNTS  OR 0/1 DF because ML models don't
    take strings!
    
    '''

    tweets_df = read_tweets_from_file(tweets_raw)
    ## Create string of tweet text
    tweets_text = " ".join(tweets_df)

    ## If phase 1, read each JSON from tweets_random file
    if tweets_random != None:
        read_tweets_from_file(tweets_raw, tweets_df)
        ## Create string of nonrelevant tweet text
        bad_tweets_text = " ".join(tweets_df)
        return tweets_df, tweets_text, bad_tweets_text

    return tweets_df, tweets_text

def read_tweets_from_file(file_name, tweets_df = []):
    tokenizer = TweetTokenizer(preserve_case = False, strip_handles = True)
    ## Read each JSON from file
    with open(file_name) as data_file:
        for tweet in data_file.readlines():
            text = json.loads(tweet).get("text", "")
            ## Remove links from text
            text = re.sub(r"http\S+", "", text)
            ## Remove handle, punctuation from tweet text
            text_words = filter(lambda x: x not in string.punctuation, tokenizer.tokenize(text))
            ## Add tweet text to list
            tweets_df.append(" ".join(text_words))
    return tweets_df

def semantic_indexing(tweets_df, tweets_text = None, bad_tweets_text = None):
    '''
    Person Responsible: Devin Munger

    + tweets_text: text of relevant
    + bad_tweets_text: text of not relevant tweetse

    Process text of tweets to produce list of keywords
    Text of not-relevant tweets might (?) be used for elimination purposes
    '''
    ## Extract keywords from tweet text corpus using TF-IDF algorithm
    tfidf = TfidfVectorizer(stop_words = "english")
    tfidf_matrix = tfidf.fit_transform(tweets_df)
    ## Get indexed list of feature keywords
    features = tfidf.get_feature_names()
    ## Get indexed list of feature weights
    weights = tfidf.idf_
    feature_weights = list(zip(weights, features))
    feature_weights.sort(reverse = True)
    ## Return sorted keywords
    return [x[1] for x in feature_weights]

def add_keywords_df(tweets_df, keywords):
    '''
    Person Responsible: Anna Hazard

    + tweets_df: Dataframe of tweets. May or may not have classifications
    + keywords: list of keywords that indicate relevance

    Check tweet text for keywords
    Add column to tweets_df for keywords contained in given tweet
    Column should contain list of strings, KEYWORDS WILL BE REPEATED IN LIST IF THEY OCCUR MORE THAN ONCE IN TWEET!
    (This is in case some weighting property is added in the future)
    Change DataFrame in place
    '''
    new_column = []

    word_list = []

    for text in tweets_df["text"]:
        for word in keywords:
            if word in text:
                word_list.append(word)
        new_column.append(word_list)

    tweets_df["keywords"] = new_column

    return


def train_model_offline(model, tweets_df, predictor_columns, classification_col):
    '''
    Person Responsible: Dani Alcala

    model: model being used for classification (will probably refer to a model in a
        dictionary of models)
    tweets_df: DataFrame of tweets which includes classifications
    predictor_columns: list of column names which are being used to predict classification

    Create and train model on tweet_df

    This is the offline function to use to determine which is the best model to use
    DO NOT RUN THIS FUNCTION with the interface, only for our purposes.

    Returns the best model according to evaluation criteria
    '''
    models_to_run = ['LR'] #add more from above

    train, test = train_test_split(tweets_df, test_size = 0.2)

    best_model = ''
    best_auc = 0
    best_params = ''

    for index,clf in enumerate([clfs[x] for x in models_to_run]):
        running_model = models_to_run[index]
        parameter_values = grid[running_model]
        for p in ParameterGrid(parameter_values):
            clf.set_params(**p)
            clf.fit(train[predictor_columns], train[classification_col])
            if hasattr(clf, 'predict_proba'):
                y_pred_probs = clf.predict_proba(test[predictor_columns])[:,1] #second col only for class = 1
            else:
                y_pred_probs = clf.decision_function(test[predictor_columns])

            AUC = evaluate_model(test, classification_col, y_pred_probs)

            if AUC > best_auc:
                best_model = running_model
                best_auc = AUC
                best_params = clf

    return best_model, best_params

def evaluate_model(test_data, classification_col, y_pred_probs):
    '''
    Evaluate model with AUC of Precision-recall curve

    DO WE WANT RECALL at a specific precision point, instead?
    '''
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(test_data[classification_col], y_pred_probs)
    precision = precision_curve[:-1]
    recall = recall_curve[:-1]

    AUC = auc(recall, precision)

    return AUC


def train_model(tweets_df, predictor_columns=[], classification_col="", best_model="NB", best_params=""):
    '''
    Given the best model and best parameters obtained from running train_model_offline,
    this function will train the best model during user interaction

    tweets_df: DataFrame of tweets which includes classifications

    '''
    clf = clfs[best_model]  
    clf.set_params(**best_params)
    model = clf.fit(tweets_df[predictor_columns], tweets_df[classification_col])

    return model 

    #NOTE: good idea - I assumed the two would be combined under predict_classification with the arguments (predictor_columns, train_df, classify_df)
    # when I called it elsewhere

def predict_classification(predictor_columns, tweets_df_unclassified, model):
    '''
    Person Responsible: Dani Alcala

    model: trained model being used for classification
    tweets_df: DataFrame of tweets which DOES NOT include classification
    predictor_columns: list of column names which are being used to predict classification
    Modify DataFrame in place 

    REVERT TO OLD PARAMS?
    '''

    predicted_values = model.predict(tweets_df_unclassified[predictor_columns])

    tweets_df_unclassified['class'] = predicted_values


def classify_tweets(tweets_df, keyword_dict):
    '''
    Person Responsible: Anna Hazard

    This is not the function where the model is used to predict classifications!

    + tweets_df: DataFrame of tweets. At this point tweets_df should have a keywords
    column, and may or may not already have a classification column.
    + keyword_dict: Dictionary containing keywords with user feedback,
    format: {"bezoar": "bad", "turquoise": "neutral", "hrc": "good" ...}

    Add or change classification based on classificatiion of keywords in tuples list
    Tentatively:
    Tweets containing "bad" words should be classified as irrelevant
    Tweets containing *only* neutral words should be classified as irrelevant
    Tweets containing "good" and "neutral" words and *no* "bad" words
    should be classified as relevant
    '''

    class_column = []

    for word_list in tweets_df["keywords"]:
        word_class_list = []
        for word in word_list:
            word_class_list.append(keyword_dict[word])
        if "bad" in word_class_list:
            classification = "irrelevant"
        elif "good" not in word_class_list:
            classification = "irrelevant"
        else:
            classification = "relevant"

        class_column.append(classification)

    tweets_df["classificatiion"] = class_column
    
    return

def keyword_binary_col(keywords, tweet_df):
    '''
    Create binary features for all keywords
    Transform other non-numeric data
    Change dataframe in place and return list of column names that should be used for model training
    '''
    predictor_columns = []

    return predictor_columns

def get_keywords(tweets_df):
    '''
    Person Responsible: Anna Hazard

    tweets_df: DataFrame of tweets with keyword column and classification

    collect keywords from tweets classified as "relevant"
    '''

    keywords = set([])

    for i, word_list in enumerate(tweets_df["keywords"]):
        if tweets_df["classification"][i] == "relevant":
            keywords.update(word_list)

    return list(keywords)

def update_keywords(keyword_dict):
    '''
    Returns list of keywords based on user feedback with bad and neutral words removed
    '''
    new_keywords = []
    for key, value in keyword_dict.items():
        if value == "good":
            new_keywords.append(key)

    return new_keywords
