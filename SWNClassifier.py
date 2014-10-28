import nltk
import yaml
import sys
import os
import re
from nltk.corpus import movie_reviews
import  collections
from iSentiWordNet import *
import matplotlib.pyplot as plt


class SWNClassifier(object):
    '''
    This class calculate a document and its sentences pos and neg scores, classify them into
    pos or neg sentences and document. Also it gives them pos and neg rates

    Given a group of files, POS files and NEG files
    It calculats POS sentence threshold, NEG sentence threshold, 
                 POS document threshold, NEG document threshold.
    Model:
        For sentences:
        The sample files just labeled document positive or negative, not each sentence. 
        We suppose for positive document, positive sentences dominate the file.
        We calculate the positive sentence frequence as follows:
        First calculate pos score range, (0, 1) SPS is the random variable, SPS in (0,1)
        suppose it sbjects to normal distribution, extimate: mean(SPS) and std(SPS)
        print out  its frequencies.
        SPSD = SPS - SNS      SPS in (0, 1); SNS in (0,1); but SPSD in (-1, 1) but we only keep POS sentences,
            that is if SPSD <0, we ignore it, therefore SPSD is in (0, 1)

        SPSD score 0.01, 0.02, ..., 0.99
        Frequency   2     5          2
        
        The same as SNS,

        Document DPS, DNS

        For a sentence
        1. Check if it is POS or NEG
            if doc lable is if sum(word pos)  > sum(word neg) :   POS
            else :  NEG

        2.    put it in SENT_POS_SCORES vector according to its score

        For document
        1. For labeled POS document
            1.1 Normalize sentence POS score
                N non zero score words
                sent pos score  = (word pos scores - word neg scores) /N
            1.2 Doc pos score
                M pos sentence
                doc pos score = sum(sent pos score)/m
            1.3 put doc pos score into  DOC_POS_SCORES vector accorting to its doc pos score
        2. Same as labeled NEG document                
    '''

    def __init__(self, interface_swn, prepressor):
        self.SENT_POS_TH = 0.01 #sentence pos threshold 
        self.SENT_NEG_TH = 0.01 #sentence neg threshold 
        self.DOC_POS_TH = 0.05 #document pos threshold
        self.DOC_NEG_TH = 0.05 #document neg threshold
        self.iswn = interface_swn
        self.pre_processor = prepressor

    def sent_scores_in_training(self, sent, pos = True) :
        '''
        give pos or neg scores to the sentence
        input: sent -- tagged sentence, 
               pos -- True, the sentence is POS one, False -- NEG one
    	
        Calculate sentence  scores
        For POS sentence, the score POS = MAX(words POS)
        For NEG			NEG = MAX(words POS)
        Determine if the sentence is positive or negative  
                   Positive: 	MAX(words POS) > MAX(words POS)
                   Negative:   MAX(words NEG)  >  MAX(words POS) 
        output: sent pos score, sent neg score
        '''
        sent_pos_score, sent_neg_score = self.cal_sent_scores(sent)
        if pos == True :
            if sent_pos_score > sent_neg_score:
                return sent_pos_score, sent_neg_score
            else :
                return 0.0, 0.0
        else :
            if sent_pos_score < sent_neg_score:
                return sent_pos_score, sent_neg_score
            else :
                return 0.0, 0.0

    # Calcule the document scores in training 
    # input:
    #       doc -- list of sentences whihc are pos tagged ones
    #       pos -- if the document is positive os negative
    # doc scores are average of the pos/neg sentences socres
    # output:  (label POS/NEG, pos_score, neg_score, [(sent_pos_score, sent_neg_score), ... ]) 
    def doc_scores_in_training(self, doc, pos = True) :
        doc_pos_score = 0.0
        doc_neg_score = 0.0
        sent_pos_score = 0.0
        sent_neg_score = 0.0
        sent_count = 0.0
        sents_scores = []
        for sent in doc :
            sent_pos_score, sent_neg_score = self.sent_scores_in_training(sent, pos)
            if sent_pos_score >0.0 or doc_neg_score > 0: 
                doc_pos_score += sent_pos_score
                doc_neg_score += sent_neg_score
                sent_count += 1
                sents_scores.append((sent_pos_score, sent_neg_score))

           
        if pos == True :
            if doc_pos_score > doc_neg_score :
                return (pos, doc_pos_score/sent_count, doc_neg_score/sent_count , sents_scores) #average socres
            else :
                return ( pos, 0.0, 0.0, sents_scores)
        if pos == False :
            if doc_pos_score < doc_neg_score :
                return (pos, doc_pos_score/sent_count, doc_neg_score/sent_count, sents_scores)
            else :
                return ( pos, 0.0, 0.0, sents_scores)

    def training(self, file_folder_path):
        #movie_reviews
        if os.path.exists(file_folder_path) == False :
            print "Error: " + file_folder_path
            return 

        #process pos files
        doc_pos_distribute = [0]*101 # pos score*100 
        sent_pos_distribute = [0]*101
        doc_neg_distribute = [0]*101
        sent_neg_distribute = [0]*101
        pos_files =  self.files_in_dir( file_folder_path, pos = "pos")
        for pos_file in pos_files:
            print pos_file
            review = self.pre_process_file(pos_file) 
            analyzed_doc = self.doc_scores_in_training(review, True) #(label POS/NEG, pos_score, neg_score, [(sent_pos_score, sent_neg_score), ... ])
            if analyzed_doc[0] == True and analyzed_doc[1] > 0.0:
                #doc score = (pos_score - neg_score)*100
                doc_pos_distribute[int((analyzed_doc[1] - analyzed_doc[2])*100+0.5)] += 1
                #calculate sentence score distribution
                for sent in analyzed_doc[3] :
                    if sent[0] > sent[1] :
                        sent_pos_distribute[int((sent[0] - sent[1])*100+0.5)] += 1

        neg_files =  self.files_in_dir( file_folder_path, pos = "neg")
        for neg_file in neg_files:
            print neg_file
            review = self.pre_process_file(neg_file) 
            analyzed_doc = self.doc_scores_in_training(review, False) #(label POS/NEG, pos_score, neg_score, [(sent_pos_score, sent_neg_score), ... ])
            if analyzed_doc[0] == False and analyzed_doc[2] > 0.0:
                #doc score = (pos_score - neg_score)*100
                doc_neg_distribute[int((analyzed_doc[2] - analyzed_doc[1])*100+0.5)] += 1
                #calculate sentence score distribution
                for sent in analyzed_doc[3] :
                    if sent[1] > sent[0] :
                        sent_neg_distribute[int((sent[1] - sent[0])*100+0.5)] += 1
        self.DOC_POS_TH = self.calculate_threshold(doc_pos_distribute)
        self.DOC_NEG_TH = self.calculate_threshold(doc_neg_distribute)
        self.SENT_POS_TH = self.calculate_threshold(sent_pos_distribute)
        self.SENT_NEG_TH = self.calculate_threshold(sent_neg_distribute)
        print self.DOC_POS_TH, "; ", self.DOC_NEG_TH, "; ", self.SENT_POS_TH,"; ",self.SENT_NEG_TH

        plt.plot(doc_pos_distribute)
        plt.ylabel('Doc POS Distributiion')
        plt.show()
        plt.plot(doc_neg_distribute)
        plt.ylabel('Doc NEG Distributiion')
        plt.show()
        plt.plot(sent_pos_distribute)
        plt.ylabel('Sent POS Distributiion')
        plt.show()
        plt.plot(sent_neg_distribute)
        plt.ylabel('Sent NEG Distributiion')
        plt.show()
        return


    # threshold x, x frequence <= 0.02  
    def calculate_threshold(self, distribution):
        total = sum(distribution)
        if total <= 0 :
            return 0.0
        frequenciess = [float(f)/total for f in distribution]
        total_refq = 0.0
        count = 0
        threshold = 0.0
        while count <= 99:
            total_refq += frequenciess[count]
            if total_refq > 0.05 :
                i = max(count-1, 0)
                threshold = float(i)/100.0
                break
            count += 1
        return threshold


    def cal_sent_scores(self, sentence):
        """
            calculate a sentence sentiment scores based on max word pos/neg score
            input: a sentence list, whose element is a tuple (word, pos tag)
            output: sentence pos, neg score
        """
        word_count = 0
        max_word_pos_score = 0
        max_word_neg_score = 0
        for word, tag in sentence:
            pos_score = 0
            neg_score = 0
            synsets = self.iswn.senti_synsets(word, tag)      
            num_synsets  =  len(synsets)   
            word_pos_score = 0
            word_neg_score = 0
            if num_synsets >=1 :               
                for synset in synsets:
                    word_pos_score += synset.pos_score
                    word_neg_score += synset.neg_score
                word_pos_score = word_pos_score/num_synsets  #average synsets scores
                word_neg_score = word_neg_score/num_synsets
            if max_word_pos_score < word_pos_score :
                max_word_pos_score = word_pos_score
            if max_word_neg_score < word_neg_score :
                max_word_neg_score = word_neg_score
        
        return max_word_pos_score, max_word_neg_score

    def is_sent_pos(self, pos_score, neg_score) :
        if pos_score - neg_score >= self.SENT_POS_TH: 
            return True
        else :
            return False

    def is_sent_neg(self, pos_score, neg_score) :
        if neg_score - pos_score >= self.SENT_NEG_TH: 
            return True
        else :
            return False

    def is_sent_neutral(self, pos_score, neg_score) :
        if pos_score - neg_score < self.SENT_POS_TH and neg_score - pos_score < self.SENT_NEG_TH :
            return True
        else :
            return False

    def classify_sent(self, pos_score, neg_score) :
        if pos_score - neg_score >= self.SENT_POS_TH: 
            return (1, (pos_score - neg_score))
        elif neg_score - pos_score >= self.SENT_NEG_TH:
            return (-1, (neg_score - pos_score))
        else :
            return (0, 0.0)


    def cal_doc_scores(self, sentences) :
        """
            calculate document pos and neg scores
            input: sentences, a list of sentences, element in the list is a tuple(label, pos_score, neg_score)
            label 1 -- pos; -1 -- neg; 0 -- neutral
            output: return pos_score, neg_score
        """
        doc_pos_score =0
        doc_neg_score = 0
        for label, pos, neg in sentences:
            if label != 0 :
                doc_pos_score += pos
                doc_neg_score += neg
        return doc_pos_score, doc_neg_score
    
      
    def classify_doc(self, pos_score, neg_score) :
        if pos_score - neg_score >= self.DOC_POS_TH: 
            return (1, (pos_score - neg_score))
        elif neg_score - pos_score >= self.DOC_NEG_TH:
            return (-1, (neg_score - pos_score))
        else :
            return (0, 0.0)      
    

    def sentiment_analysis(self, sentences):
        '''
        analyze the whole document and its sentences sentiment rates
        input: list of sentences splitted to words and their pos tags
        output: (doc_label, doc_score, list of sentences with their sentenment scores)
        '''
        doc_pos_score = 0
        doc_neg_score = 0
        sents_results = []
        count = 0
        for sent in sentences:
            sent_pos_score, sent_neg_score = self.cal_sent_scores(sent)
            doc_pos_score += sent_pos_score
            doc_neg_score += sent_neg_score
            sents_results.append(self.classify_sent(sent_pos_score, sent_neg_score))
            count += 1
        doc_pos_score = doc_pos_score/count
        doc_neg_score = doc_neg_score/count
        
        return self.classify_doc(doc_pos_score, doc_neg_score)
        
       
    def evaluate_doc(self, aspects) :
        """
            Calculate document scores according to the aspects scores
            input: aspects - type dictionary, its element: key - string, aspect; value - list of [frequency, pos_score, neg_score]
            return (pos_neg_label, score)   pos_neg_label: 1 - positive, -1 - negative, 0 - nutral
                                            score - the document pos/neg score
        """
        pos_score = 0
        neg_score = 0
        for key, value in aspects.items() :
            pos_score += value[1]/value[0] #average aspect score
            neg_score += value[2]/value[0]
        count = len(aspects)
        pos_scores = pos_score/count
        neg_score = neg_score/count
        return self.classify_doc(pos_scores, neg_score)

    def read_book_reviews(self, file_name) :
        file = open(file_name, 'r')
        self.reviews = file.read()
        return self.reviews


