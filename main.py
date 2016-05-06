import os
from twython import Twython 
from twython import TwythonStreamer
import argparse
import twitter
import json
import os.path
import signal
import sys

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

## FUNCTION TO QUERY TWITTER API
# should output a json of Tweets for now, or store them

class AuOth:
    def __init__(self, CK, CKS, AT, ATS):
        self.ConsumerKey = CK
        self.ConsumerKeySecret = CKS
        self.AccessToken = AT
        self.AccessTokenSecret = ATS

    def getConsumerKey(self):
        return self.ConsumerKey

    def getConsumerKeySecret(self):
        return self.ConsumerKeySecret

    def getAccessToken(self):
        return self.AccessToken

    def getAccessTokenSecret(self):
        return self.AccessTokenSecret

class MyStreamer(TwythonStreamer):
    def on_success(self, data):
        if 'text' in data:
            print(data['text'].encode('utf-8'))

    def on_error(self, status_code, data):
        print(status_code)

        # Want to stop trying to get data because of the error?
        # Uncomment the next line!
        # self.disconnect()

def makeAuth(infoStr):
    CK, CKS, AT, ATS = infoStr.split(",")
    return AuOth(CK, CKS, AT, ATS)

def get_creds(filename):
    infile = open(filename, 'r')
    c = makeAuth(infile.readline())
    return c


def get_tweets_from_stream(filter_, num_tweets, auth):
  
    # Connect to the stream
        twitter_stream = twitter.TwitterStream(auth=auth)
    
        if filter_ is None and filters_file is None:
            stream = twitter_stream.statuses.sample()
        else:
            if filter_ is not None:
                track = filter_
            elif filters_file is not None:
                track = ",".join(filters_file.read().strip().split("\n"))

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
            # The public stream includes tweets, but also other messages, such
            # as deletion notices. We are only interested in the tweets.
            # See: https://dev.twitter.com/streaming/overview/messages-types
            if tweet.has_key("text"):
                # We also only want English tweets
                if tweet["lang"] == "en":
                    save_tweet(tweet, outf, format)
                    fetched += 1
                    if fetched % 100 == 0:
                        now = datetime.now().isoformat(sep=" ")
                        msg = "[{}] Fetched {:,} tweets.".format(now, fetched)
                        if outf != sys.stdout: print(msg)
                    if num_tweets > 0 and fetched >= num_tweets:
                        break
    


creds = get_creds('secrets.txt')
consumer_key, consumer_secret, oauth_token, oauth_secret =  creds.ConsumerKey, creds.ConsumerKeySecret, creds.AccessToken,creds.AccessTokenSecret
auth = twitter.OAuth(oauth_token, oauth_secret, consumer_key, consumer_secret)
print oauth_token
print oauth_secret
print consumer_key
print consumer_secret
#t = twitter.Twitter(auth=auth)
users = ['aurelionuno']
twitter_stream = twitter.TwitterStream(auth=auth)
iterator = twitter_stream.statuses.sample()

tweet_count = 10
for tweet in iterator:
  tweet_count -= 1
  print json.dumps(tweet)  

# if __name__=="__main__":
#     instructions = '''Usage: main.py word 
#         '''

#     if(len(sys.argv) != 2):
#         print(instructions)
#         sys.exit()

#     word = sys.argv[1]

#     # QUERY TWITTER WITH WORD, get list of words back

#     fake_words = ['hilary', 'prez', 'shower']
#     response_dict = {}

#     print '''WELCOME TO TWITTER DETECTIVE! \n 
#                     The word you selected to begin your query is:  %s ''' % word
#     for w in fake_words:
#         response = raw_input('Enter 1 if {} is a word associated with your word / topic of interest: '.format(color.BOLD + w + color.END))
#         response_dict[w] = response

#     print response_dict
#     wd = os.getcwd()

