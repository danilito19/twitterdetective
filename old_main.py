import os
import argparse
import json
import os.path
import signal
import sys
import cmd
import functions as fct

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

if __name__=="__main__":

    instructions = '''Usage: main.py word 
         '''

    if(len(sys.argv) != 2):
        print(instructions)
        sys.exit()

    words = sys.argv[1]

    # QUERY TWITTER WITH WORD, get list of words back

    fake_words = ['hilary', 'prez', 'shower']
    response_dict = {}

    print '''WELCOME TO TWITTER DETECTIVE! \n 
             The word you selected to begin your query is:  %s ''' % word
    for w in fake_words:
    response = raw_input('Enter 1 if {} is a word associated with your word / topic of interest: '.format(color.BOLD + w + color.END))
    response_dict[w] = response

    print response_dict
    wd = os.getcwd()