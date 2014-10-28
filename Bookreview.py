from collections import defaultdict
from Preprocessor import *
from iSentiWordNet import *
from SWNClassifier import *
from Aspect import *


class Bookreview(object):
    """
        This class will be GUI class, currently, it just performs high level control and
        forwards the requirements to other objects
    """
    def __init__(self, swn_filename, working_dir, mco_file):
        self.reviews = ""
        self.pre_processor = Preprocessor() 
        self.iswn = SentiWordNetCorpusReader(swn_filename)
        self.swn_classifier = SWNClassifier(self.iswn, self.pre_processor)
        self.aspect = Aspect(self.iswn, self.pre_processor, working_dir, mco_file)
        self.aspects = defaultdict() #element: key(aspect), [frequency, pos_score, neg_score]
        #self.aspects["test"].append([1, 0.5, 0.2])

    def do_analysis(self, file_path):
        #Get a simplifed tagged and non-simplifed tagged sentences 
        simplifed_pos_sentences, pos_sentences = self.pre_processor.pre_process_file(file_path)
        doc_results = (0, 0.0)
        #sentiment analysis, determine document opinion oritantation
        doc_results = self.swn_classifier.sentiment_analysis(simplifed_pos_sentences) 
        # Extract aspects
        aspects = defaultdict() #aspects is a dictionary of the element: key, [frequency, pos_score, neg_score ]
        for idx, sent in enumerate(pos_sentences) :
            parsed_sent = self.aspect.sent_parse(sent)
            #Extract sentiment words from a sentence according to the pos/neg scores from SentiWordNet
            sentiment_words = self.aspect.extract_sentiment_words(simplifed_pos_sentences[idx])
            if len(sentiment_words) >= 1 :
                #Extract aspects from the sentence based on the sentiment word
                sent_aspect = self.aspect.extract_aspects(parsed_sent, sentiment_words)
                if sent_aspect is None :
                    pass
                else :
                    if len(sent_aspect) > 0 :
                        first_aspect = sent_aspect[0]
                        if first_aspect is not None :
                            key = first_aspect[1]
                            if len(self.aspects) == 0 or self.aspects.has_key(key) ==False  :
                                aspect = [1, first_aspect[2], first_aspect[3]]
                                self.aspects[key] = aspect
                            else :
                                value = self.aspects[key]
                                value[0] += 1
                                value[1] += first_aspect[2]
                                value[2] += first_aspect[3]
                        else :
                            pass
                    else :
                        pass
        # Calculate document pos and neg scores according to the aspects scores
        doc_results = self.swn_classifier.evaluate_doc(self.aspects)
        return doc_results, self.aspects


     
