import os
import argparse
import json
import os.path
import signal
import sys
from twitterlock import Twitterlock

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

    intro = color.GREEN + "Welcome to TwitterDetective!" + color.END
    print(intro)
    print("Please type a term or terms to begin building your query. If using multiple terms separate with a space only")


    first_query = input(color.BOLD + "Query terms: " + color.END)
    # makes raw input into array passed to twitterlock ['hilary', 'clinton']
    query_words = first_query.split()


    tw = Twitterlock(words = query_words)

    tw.cycle1()

    print(color.BOLD + "Here are your suggested search terms:" + color.END)

    print(", ".join(tw.keywords))

    cont = input("Are you satisfied with this list (y/n)? ")

    if cont == "y":

      tw.set_satisfaction(True)

    while not tw.satisfactory:

        print("Indicate the relevance of each term to your interest.")

        response_dict = {1: "good", "relevant": "good", 2: "neutral", "neutral": "neutral", 3:"bad", "irrelevant": "bad"}
        feedback = {}
        for word in tw.keywords:
            prompt = word + " is (1) relevant, (2) neutral, (3) irrelevant: "
            response = input(prompt)
            feedback[word] = response

        tw.take_feedback(feedback)

        print("Here are your suggested search terms:")

        print(", ".join(tw.keywords))

        cont = input("Are you satisfied with this list (y/n)? ")

        if cont == "n":

            tw.cycle2()

        elif cont == "y":

            tw.set_satisfaction(True)

    filename = input("Where would you like the results of your query stored? Type file path: ")

    tw.finish(filename)


