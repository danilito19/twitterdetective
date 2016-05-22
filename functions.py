
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
from sklearn.cross_validation import train_test_split, KFold


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

def cycle1(word_list):
    for word in word_list:
        print(word)
    return ["alpha", "beta", "theta"], None

def cycle2(feedback_dict, tweet_df):
    return ["crimson", "calico", "qwark"], None

def build_query(query_words, exclude_words = None):
    '''
    Person Responsible: Manu Aragones

    + query_words: list of words / phrases to be included in query
    + exclude_words: list of words to exclude from results (should
        this be here or in later function? Depends if API allows exclusion)
    
    builds query for twitter API from user input
    '''
    return query

def get_tweets(query, size):
    '''
    Person Responsible: Manu Aragones

    + query: string, query for twitter API
    + size: number of tweets desired
    + other arguments?

    Takes query, queries twitter API, returns JSON of tweets

    If query is None, random sample of tweets should be selected
    '''
    return tweets_raw

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
    '''
    return tweets_df, tweets_text, bad_tweets_text
    return tweets_df, tweets_text

def semantic_indexing(tweets_text, bad_tweets_text = None):
    '''
    Person Responsible: Devin Munger

    + tweets_text: text of relevant
    + bad_tweets_text: text of not relevant tweetse

    Process text of tweets to produce list of keywords
    Text of not-relevant tweets might (?) be used for elimination purposes
    '''
    return keywords

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

def train_model_offline(tweets_df, model='NB' predictor_columns=[], classification_col=""):
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

    return model #could combine func with predict_classification func (note - good idea, I have just used "predict_classification" in the twitterlock class assuming this would be done)

def predict_classification(model, tweets_df_unclassified, predictor_columns):
    '''
    Person Responsible: Dani Alcala

    model: trained model being used for classification
    tweets_df: DataFrame of tweets which DOES NOT include classification
    predictor_columns: list of column names which are being used to predict classification
    Modify DataFrame in place 
    '''

    predicted_values = model.predict(tweets_df_unclassified[predictor_columns])

    tweets_df_unclassified['class'] = predicted_values

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



