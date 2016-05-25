class autorizador:
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

def makeAuth(string_):
    # extracts credentials from string and returns autorizador class object
    CK, CKS, AT, ATS = string_.split(",")
    return autorizador(CK, CKS, AT, ATS)

def get_creds(filename):
    # open the input file for reading
    infile = open(filename, 'r')
    c = makeAuth(infile.readline())
    for line in infile:
        s = makeAuth(line)
    infile.close()
    return c