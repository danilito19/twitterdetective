##########

# MAIN PROGRAM FILE

##########


## FUNCTION TO QUERY TWITTER API
# should output a json of Tweets for now, or store them








if __name__=="__main__":
    instructions = '''Usage: main.py word 
        '''

    if(len(sys.argv) != 2):
        print(instructions)
        sys.exit()

    word = sys.argv[1]

    #take user's word and query Twitter API function
