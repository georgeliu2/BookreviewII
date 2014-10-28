import nltk
import yaml
import sys
import os
import re
import  collections
from os.path import isfile, join
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import movie_reviews

class Preprocessor(object):
    '''
    Pre-process plain text files, split it into sentences, tokenlize them into tokens,
    and label part of speech tags for tokens
    '''
    def __init__(self):
        self.nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def pre_process_file(self, filepath):
        fileobj = open(filepath, 'r')
        file_str = fileobj.read()
        #split into sentences, tokenlize each sentence and then pos tag them
        splited_sentences = self.split(file_str)
        return self.pos_tag_sents(splited_sentences) 
    
    def split(self, text):
        """
        input format: a paragraph of plain text
        output format: a list of lists of words.
            e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
        """
        sentences = self.nltk_splitter.tokenize(text)  #split text into sentences in a list        
        tokenized_sentences = [self.nltk_tokenizer.tokenize(sent) for sent in sentences] 
        return tokenized_sentences

    def pos_tag_sents(self, sentences) :
        '''
        analyze a sentence, and add pos tags to words
        input: a list whose element is a sentence list whose elements are terms splitted from the sentence
        output: return two lists, a simplified pos tagged sentence and not simplifed tagged sentence
                an element of the list  a tuple (word, pos tag) 
        '''
        tagged_sents = []
        simp_tagged_sents = []
        for sent in sentences:
            tagged_sent = nltk.pos_tag(sent)
            tagged_sents.append(tagged_sent)
            simplified = [(word, self.get_wordnet_pos(tag)) for word, tag in tagged_sent]
            simp_tagged_sents.append(simplified)
        return simp_tagged_sents, tagged_sents


    def get_wordnet_pos(self, treebank_tag):
        """
            convert TreeBank TAGs to WordNet TAGs
        """
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('R'):
            return wn.ADV
        else:
            return ''


