from Preprocessor import *
from iSentiWordNet import *
from SWNClassifier import *

SWN_FILENAME = "C:\\Python27\\nltk_data\\corpora\\sentiwordnet\\SentiWordNet_3.0.0.txt"

class Bookreview(object):
    # This will be GUI class, currently, no any functionality
    def __init__(self):
        self.reviews = ""
        self.pre_processor = Preprocessor() 
        self.iswn = SentiWordNetCorpusReader(SWN_FILENAME)
        self.swn_classifier = SWNClassifier(self.iswn, self.pre_processor)

    def do_analysis(self, file_path):
        sentences = self.pre_processor.pre_process_file(file_path)
        sents = []
        doc_results = (0, 0.0)
        doc_results, sents = self.swn_classifier.sentiment_analysis(sentences) #sentiment analysis, determine document opinion
        return doc_results, sents


     
