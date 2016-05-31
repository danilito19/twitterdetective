import os
from twython import Twython 
from twython import TwythonStreamer
import argparse
import twitter
import json
import os.path
import signal
import sys
from autorizador import * 
import datetime

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
    track = filter_words
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
        # The public stream includes tweets, but also other messages, such
        # as deletion notices. We are only interested in the tweets.
        # See: https://dev.twitter.com/streaming/overview/messages-types
        if tweet.get('text'):
            # We also only want English tweets
            if tweet['lang'] == "en":
                save_tweet(tweet, outf)
                fetched += 1
                # if fetched % num_tweets == 0:
                #     # now = datetime.now().isoformat(sep=" ")
                #     # msg = "[{}] Fetched {:,} tweets.".format(now, fetched)
                #     if outf != sys.stdout: 
                #         print(msg)
                #         stream.close()
                if num_tweets > 0 and fetched >= num_tweets:
                    stream.close()
                    break

get_tweets('Trump', 20, 'TEST')
