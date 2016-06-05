import os
import argparse
import json
import os.path
import signal
import sys
from twitterlock import Twitterlock
from six.moves import input #for python2.7


class color:
    '''
    use to make terminal interfase pretty.
    usage: 
    print color.BOLD + 'Hello World !' + color.END
    '''

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


if __name__ == "__main__":

    intro = color.GREEN + "\nWelcome to TwitterDetective!" + color.END
    print(intro)

    print(color.DARKCYAN + "\nName this session\n" + color.END)
    temp_file = [input(color.BOLD + "Session Name: " +color.END)]
    temp_file_processed = "Sessions/"+  "_".join(temp_file) + ".txt"
    print(color.DARKCYAN + "\nDefine session scope\n" + color.END)

    num_tweets = input(color.BOLD + "Number of Tweets: " +color.END)

    num_tweets_process = int(num_tweets)

    print(color.DARKCYAN + "\nPlease type a term or terms to begin building your query.\nIf using multiple terms separate with a space only\n" + color.END)

    first_query = input(color.BOLD + "Query terms: " + color.END)
    query_words = first_query.split()

    # query_words = ['hilary']
    # num_tweets_process = 50
    # temp_file_processed = 'Sessions/d.txt'
    tw = Twitterlock(words = query_words, size = num_tweets_process, filename= temp_file_processed)

    tw.cycle1()

    while not tw.satisfactory:

        print(color.BOLD + "Here are some additional search terms:" + color.END)

        print(", ".join(tw.keywords))

        cont = input("Are you satisfied with all these terms (y/n)? ")

        if cont == "y":

            tw.set_satisfaction(True)

        else:

            print("Indicate the relevance of each term to your interest.")

            response_dict = {1: "good", "relevant": "good", 2: "neutral", "neutral": "neutral", 3:"bad", "irrelevant": "bad"}
            feedback = {}
            for word in tw.keywords:
                prompt = color.PURPLE + word + color.END + " is (1) relevant, (2) neutral, (3) irrelevant: "
                response = input(prompt)
                feedback[word] = int(response)
            tw.take_feedback(feedback)

            print("Here is your full list of suggested search terms:")

            print(", ".join(tw.master_keywords))

            cont = input("Are you satisfied with this list (y/n)? ")

            if cont == "n":

                tw.cycle2()

            elif cont == "y":

                tw.set_satisfaction(True)

    filename = input("Where would you like the results of your query stored? Type file path: ")

    tw.finish(filename)


