from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, ensemble
from sklearn.metrics import *
from sklearn.cross_validation import train_test_split, KFold
from autorizador import *
import twitter
import sys
import json
import string
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import datetime
import pandas as pd
import random
import csv
from sklearn.grid_search import ParameterGrid
import matplotlib.pyplot as plt
import pylab
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

plt.rcParams["figure.figsize"] = [18.0, 8.0]



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
'SVM' :{'C' :[0.001,0.01,0.1,1], 'penalty': ['l1', 'l2']},
'KNN' :{'n_neighbors': [1, 3, 5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
       }

MODELS_TO_RUN = ['LR', "NB", 'SVM', "RF"] #add more from above
# BEST_MODEL = "NB"
# BEST_PARAMS = ''


def get_credents():
	creds = get_creds('secrets-manu.txt')
	auth=twitter.OAuth(creds.AccessToken,creds.AccessTokenSecret,creds.ConsumerKey, creds.ConsumerKeySecret)
	return auth

def save_tweet(tweet, f):
    # Replace HTML entities; function extracted from Borja Sotomayor Twitter Harvester 
    tweet['text'] = tweet['text'].replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
    json.dump(tweet, f)
    f.write('\n')

def save_user_tweets(user, n, auth):
    """
    Function that iterates through every tweet in the stream and calls the save_tweet function to 
    dump each tweet in a text file in json format
    """
    t = twitter.Twitter(auth=auth)
    print("Fetching %i tweets from @%s" % (n, user))
    tweets = t.statuses.user_timeline(screen_name=user, count=n)
    print("  (actually fetched %i)" % len(tweets))
    for tweet in tweets:
        save_tweet(tweet, outfile)


def get_tweets(filter_words, num_tweets, filename):
    '''
    Person Responsible: Manu Aragones

    + filter_file: list of words / phrases to be included in query IN FILE
    + filter_words: if filter_file not specified -> you can input filter manually
                    format: comma-separated list of phrases which will be used to
                            determine what Tweets will be delivered on the stream
    + num_tweets: the maximum number of tweets before break
    builds query for twitter API from user input

    Simplified fuction from Borja Sotomayor Twitter Harvester

    '''
    outf = open(filename, "w")
    # Connect to the stream
    auth = get_credents()
    twitter_stream = twitter.TwitterStream(auth=auth)
    # if filter_words is None:
    #     stream = twitter_stream.statuses.sample()
    # else:
    #     if filter_words is not None:
    track = ",".join(filter_words)
    stream = twitter_stream.statuses.filter(track=track)
    
    fetched = 0

    if num_tweets > 0:
        if outf != sys.stdout: print("Fetching %i tweets... " % num_tweets)
    else:
        signal.signal(signal.SIGINT, signal_handler)
        now = datetime.now().isoformat(sep=" ")
        msg = "[{}] Fetching tweets. Press Ctrl+C to stop.".format(now)
        if outf != sys.stdout: print(msg)

    for tweet in stream:
        if tweet.get('text'):
            # We  only want English tweets
            if tweet['lang'] == "en":
                save_tweet(tweet, outf)
                fetched += 1
                if num_tweets > 0 and fetched >= num_tweets:
                    stream.close()
                    break


def cycle1(word_list):
    for word in word_list:
        print(word)
    return ["alpha", "beta", "theta"], None

def cycle2(feedback_dict, tweet_df):
    return ["crimson", "calico", "qwark"], None

def process_tweets(file_name):
    '''
    Person Responsible: Devin Munger

    + file_name: filename of tweets as returned from API based on query
   
    Extract text from file; return dataframe with tweet text, id
    '''
    ## Create empty dataframe
    tweet_df = pd.DataFrame(columns = ["text", "id"])

    tokenizer = TweetTokenizer(preserve_case = False, strip_handles = True)
    ## Read each JSON from file
    with open(file_name) as data_file:
        for entry in data_file.readlines():
            tweet = json.loads(entry)
            tweet_id = str(tweet.get("id", ""))
            text = tweet.get("text", "")
            ## Remove links from text
            text = re.sub(r"http\S+", "", text)
            ## Remove twitter keywords
            text.replace("rt", "")
            ## Remove handle, punctuation from tweet text
            text_words = filter(lambda x: x not in string.punctuation, tokenizer.tokenize(text))
            ## Add tweet to dataframe
            tweet_df.loc[len(tweet_df)] = [" ".join(text_words), tweet_id]
    return tweet_df


def semantic_indexing(tweet_df, max_keywords = 20):
    '''
    Person Responsible: Devin Munger

    + tweets_df: dataframe containing tweet text, ids
    + max_keywords: maximum number of keywords to return

    Process text of tweets to produce list of keywords
    '''
    ## Extract keywords from tweet text corpus using TF-IDF algorithm
    tfidf = TfidfVectorizer(stop_words = "english", smooth_idf = False)
    tfidf_matrix = tfidf.fit_transform(tweet_df["text"].values)
    ## Get indexed list of feature keywords
    features = tfidf.get_feature_names()
    ## Get indexed list of feature weights
    weights = tfidf.idf_
    ## Extract highest weighted keywords
    max_weight = max(weights)
    weighted_words = [features[i] for i, x in enumerate(weights) if x == max_weight]
    ## Return sample of highest weighted keywords
    indices = random.sample(range(len(weighted_words)), min(max_keywords, len(weighted_words)))
    return [weighted_words[i] for i in indices]
    

def add_keywords_df(tweet_df, keywords):
    '''
    Person Responsible: Anna Hazard

    + tweet_df: Dataframe of tweets. May or may not have classifications
    + keywords: list of keywords that indicate relevance

    Check tweet text for keywords
    Add column to tweet_df for keywords contained in given tweet
    Column should contain list of strings, KEYWORDS WILL BE REPEATED IN LIST IF THEY OCCUR MORE THAN ONCE IN TWEET!
    (This is in case some weighting property is added in the future)
    Change DataFrame in place
    '''
    new_column = []

    for text in tweet_df["text"]:
        word_list = []
        for word in keywords:
            if word in text:
                word_list.append(word)
        new_column.append(word_list)

    tweet_df["keywords"] = new_column

    return

def train_model_offline(tweet_df, predictor_columns):
    '''
    Person Responsible: Dani Alcala

    model: model being used for classification (will probably refer to a model in a
        dictionary of models)
    tweet_df: DataFrame of tweets which includes classifications
    predictor_columns: list of column names which are being used to predict classification

    Create and train model on tweet_df

    This is the offline function to use to determine which is the best model to use
    DO NOT RUN THIS FUNCTION with the interface, only for our purposes.

    Returns the best model according to evaluation criteria
    '''
    print("TRAINING OFFLINE")
    BEST_MODEL = ''
    BEST_PARAMS = ''
    best_auc = 0

    table_file = open('parameters-table.csv', 'wb')
    w = csv.writer(table_file, delimiter=',')
    w.writerow(['MODEL', 'PARAMETERS', 'AUC'])

    train, test = train_test_split(tweet_df, test_size = 0.2)


    for index,clf in enumerate([clfs[x] for x in MODELS_TO_RUN]):
        running_model = MODELS_TO_RUN[index]
        parameter_values = grid[running_model]
        for p in ParameterGrid(parameter_values):
            clf.set_params(**p)
            clf.fit(train[predictor_columns], train['classification'])
            if hasattr(clf, 'predict_proba'):
                y_pred_probs = clf.predict_proba(test[predictor_columns])[:,1] #second col only for class = 1
            else:
                y_pred_probs = clf.decision_function(test[predictor_columns])

            AUC = evaluate_model(test['classification'], y_pred_probs)
            w.writerow([running_model, clf, AUC])

            if AUC > best_auc:
                BEST_MODEL = running_model
                best_auc = AUC
                BEST_PARAMS = p

    if best_auc == 0:
        print('WARNING:  BEST AUC IS : ', best_auc)
    table_file.close()

    return BEST_MODEL, BEST_PARAMS

def evaluate_model(test_data_classification_col, y_pred_probs):
    '''
    Evaluate model with AUC of Precision-recall curve

    DO WE WANT RECALL at a specific precision point, instead?
    '''
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(test_data_classification_col, y_pred_probs)
    precision = precision_curve
    recall = recall_curve
    AUC = auc(recall, precision)

    return AUC


def predict_classification(predictor_columns, tweet_df_classified, tweet_df_unclassified, best_model, best_params, plot=False):
    '''
    Person Responsible: Dani Alcala

    tweet_df: DataFrame of tweets which DOES NOT include classification
    predictor_columns: list of column names which are being used to predict classification
    Modify DataFrame in place 

    train the model on the first dataframe using the specified columns and then 
    predict the classifictions on the second dataframe, 
    and add the classifications to this dataframe in place

    '''
    # global BEST_MODEL
    # global BEST_PARAMS
    print('PREDICTING CLASSIFICATION')

    print('BEST MODEL: ', best_model)
    print('BEST PARAMS: ', best_params)

    clf = clfs[best_model] 
    params = best_params 
    clf.set_params(**best_params)
    model = clf.fit(tweet_df_classified[predictor_columns], tweet_df_classified["classification"])

    predicted_values = model.predict(tweet_df_unclassified[predictor_columns])

    tweet_df_unclassified['classification'] = predicted_values

    #if plot parameters is True, get y_pred probs
    if plot:
        if hasattr(clf, 'predict_proba'):
            y_pred_probs = clf.predict_proba(tweet_df_unclassified[predictor_columns])[:,1] #second col only for class = 1
        else:
            y_pred_probs = clf.decision_function(tweet_df_unclassified[predictor_columns])

        plot_precision_recall(tweet_df_unclassified['classification'], y_pred_probs, best_model, best_params)


def plot_precision_recall(y_true, y_prob, model_name, model_params):

    '''
    Plot a precision recall curve for one model with its y_prob values.
    '''

    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_prob)
    precision = precision_curve[:-1]
    recall = recall_curve[:-1]
    plt.clf()
    plt.plot(recall, precision, label='%s' % model_params)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title("Precision Recall Curve for %s" %model_name)
    plt.savefig(model_name)
    plt.legend(loc="lower right")
    #plt.show()


def plot_precision_and_recall(y_true, y_prob, model_name, model_params):
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_prob)
    precision = precision_curve[:-1]
    recall = recall_curve[:-1]

    num = len(y_prob)
    pct_above_per_thresh = []
    for value in pr_thresholds:
        num_above_thresh = len(y_prob[y_prob >= value])
        pct_above_thresh = num_above_thresh / float(num)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, "blue", label = '%s' % model_param)
    ax1.set_xlabel("Percent of Population")
    ax1.set_ylabel("Precision", color = "blue")
    
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, "red")
    ax2.set_ylabel("Recall", color = "red")
    plt.title("Precision Recall Curve for %s" %model_name)
    plt.legend(loc="lower right")
    #plt.show()


def classify_tweets(tweet_df, keyword_dict):
    '''
    Person Responsible: Anna Hazard

    This is not the function where the model is used to predict classifications!

    + tweet_df: DataFrame of tweets. At this point tweet_df should have a keywords
    column, and may or may not already have a classification column.
    + keyword_dict: Dictionary containing keywords with user feedback,
    format: {"bezoar": "bad", "turquoise": "neutral", "hrc": "good" ...}

    Add or change classification based on classification of keywords in tuples list
    Tentatively:
    Tweets containing "bad" words should be classified as irrelevant
    Tweets containing *only* neutral words should be classified as irrelevant
    Tweets containing "good" and "neutral" words and *no* "bad" words
    should be classified as relevant
    '''
    class_column = []

    for word_list in tweet_df["keywords"]:
        word_class_list = []
        for word in word_list:
            word_class_list.append(keyword_dict[word])
        if 3 in word_class_list:
            classification = 0
        elif 1 not in word_class_list:
            classification = 0
        else:
            classification = 1

        class_column.append(classification)

    tweet_df["classification"] = class_column
    
    return

def keyword_binary_col(keywords, tweet_df):
    '''
    Create binary features for all keywords
    Transform other non-numeric data
    Change dataframe in place and return list of column names that should be used for model training
    '''
    key_dict = {}

    for word in keywords:
        key_dict[word] = []

    for word, bin_col in key_dict.items():
        # for index, row in tweet_df.iterrows():   
        #     if word in row['keywords']:
        for field in tweet_df["keywords"]:
            if word in field:
                bin_col.append(1)
            else:
                bin_col.append(0)

    for key, val in key_dict.items():
        tweet_df[key] = val

    return keywords

def get_keywords(tweet_df):
    '''
    Person Responsible: Anna Hazard

    tweet_df: DataFrame of tweets with keyword column and classification

    collect keywords from tweets classified as "relevant"
    '''

    keywords = set([])

    for i, word_list in enumerate(tweet_df["keywords"]):
        if tweet_df["classification"][i] == 1:
            keywords.update(word_list)

    return list(keywords)

def update_keywords(keyword_dict):
    '''
    Returns list of keywords based on user feedback with bad and neutral words removed
    '''
    new_keywords = []
    for key, value in keyword_dict.items():
        if value == 1:
            new_keywords.append(key)

    return new_keywords
