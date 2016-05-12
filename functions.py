
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
	Column should contain list of strings
	Change DataFrame in place
	'''
	return

def validate_keywords(keywords):
	'''
	Person Responsible: Anna Hazard

	+ keywords: list of keywords extracted from relevant tweets

	Present keywords to user for validation
	Returns list of tuples with keyword and user feedback
	User feedback can be "good", "neutral", "bad"
	Output format: [("bezoar", "bad"), ("turquoise", "neutral"), ("hrc", "good") ...]
	'''
	return keyword_tuples

def classify_tweets(tweets_df, keyword_tuples):
	'''
	Person Responsible: Anna Hazard

	¡This is not the function where the model is used to predict classifications!

	+ tweets_df: DataFrame of tweets. At this point tweets_df should have a keywords
	column, and may or may not already have a classification column.
	+ keyword_tuples: List of tuples containg keywords with user feedback,
	format: [("bezoar", "bad"), ("turquoise", "neutral"), ("hrc", "good") ...]

	Add or change classification based on classificatiion of keywords in tuples list
	Tentatively:
	Tweets containing "bad" words should be classified as irrelevant
	Tweets containing *only* neutral words should be classified as irrelevant
	Tweets containing "good" and "neutral" words and *no* "bad" words
	should be classified as relevant
	'''
	return

def train_model(model, tweets_df, predictor_columns, classification_col):
	'''
	Person Responsible: Dani Alcala

	model: model being used for classification (will probably refer to a model in a
		dictionary of models)
	tweets_df: DataFrame of tweets which includes classifications
	predictor_columns: list of column names which are being used to predict classification

	Create and train model on tweet_df
	'''
	models_to_run = ['LR']

    train, test = train_test_split(tweets_df, test_size = 0.2)

    for index,clf in enumerate([clfs[x] for x in models_to_run]):
        running_model = models_to_run[index]
        parameter_values = grid[running_model]
        for p in ParameterGrid(parameter_values):

	return model

def evaluate_model():
	'''
	need to pick from best models?
	'''
	pass

def predict_classification(model, tweets_df, predictor_columns):
	'''
	Person Responsible: Dani Alcala

	model: trained model being used for classification (will probably refer to a model in a
		dictionary of models)
	tweets_df: DataFrame of tweets which does not include classification
	predictor_columns: list of column names which are being used to predict classification
	Modify DataFrame in place
	'''
	return

def get_keywords(tweets_df):
	'''
	Person Responsible: Anna Hazard

	tweets_df: DataFrame of tweets with keyword column and classification

	collect keywords from tweets classified as "relevant"
	'''
	return