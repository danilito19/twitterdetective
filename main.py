import os
import argparse
import json
import os.path
import signal
import sys
from twitterlock import Twitterlock

if __name__ == "__main__":
    intro = "Welcome to TwitterDetective!"
    print(intro)
    print("Please type a term or terms to begin building your query. If using multiple terms separate with a space only")

    first_query = input("Query terms: ")
    query_words = first_query.split()
    tw = Twitterlock(words = query_words)

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


