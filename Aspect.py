import  nltk
import  nltk.parse

from collections import defaultdict

class Aspect(object):
    """
        It incapslates aspects related functions
        It parsers sentences, extracts sentiment words and scores, and  extracts 
        aspects and scores based on sentiment words and sentence structure 
    """
    _iswn = None
    _pre_processor = None
    _malt_parser = None
    _working_dir = ""
    _mco = ""
    _POS_NOUNS = ['NN',  #	Noun, singular or mass
                 'NNS',	#Noun, plural
                 'NNP', #	Proper noun, singular
                 'NNPS'] #	Proper noun, plural

    def __init__(self, interface_swn, prepressor, workingdir, mco_file) :
        self._iswn = interface_swn
        self._pre_process = prepressor
        self._working_dir = workingdir
        self._mco = mco_file
        self._malt_parser = nltk.parse.malt.MaltParser(working_dir=self._working_dir, #"c:\\maltparser172\\test",
        mco=self._mco)  #"engmalt.linear-1.7")

        pass
    def init_parser(self) :
        self.malt_parser = nltk.parse.malt.MaltParser(working_dir=_working_dir, #"c:\\maltparser172\\test",
            mco=_mco) #"engmalt.linear-1.7")
        pass

    def sent_parse(self, tagged_sent):
        """ Parse a sentence
            input: tagged_sent [(W1, TAG1), (W2, TAG2), ... ]
            return  [{'address': i, 'deps': [j], 'rel': 'DEP', 'tag': 'POS', 'word': WORD}, ... ]
        """
        return self._malt_parser.tagged_parse(tagged_sent)

    def extract_sentiment_words(self, parsed_sent) :
        """ Extract semtiment words from the parsed_sent
            input: parsed_sent as output of sent_parse()
                   It is a list of tuple. each element is a tuple (word, tag).
            return: a list of sentiment words [[index, 'word', pos_score, neg_score], ... ]
                Currently, it takes only one sentiment word which has the max score
        """
        #  def cal_sent_scores(self, sentence):
        words = []
        max_pos_sent_word = [-1, "", 0.0, 0.0] # (index, sentiment word, pos score, neg scor)
        max_neg_sent_word = [-1, "", 0.0, 0.0]

        for idx, (word, tag) in enumerate(parsed_sent) :
            if word =="" or word is None or tag=='TOP' :
                continue         
            word_pos_score, word_neg_score = self._iswn.get_word_score(word, tag) 
           
            if word_pos_score > 0 or word_neg_score > 0 :
                if word_pos_score >= word_neg_score :
                    if word_pos_score> max_pos_sent_word[2] :
                        max_pos_sent_word[0] = idx
                        max_pos_sent_word[1] = word
                        max_pos_sent_word[2] = word_pos_score
                        max_pos_sent_word[3] = word_neg_score
                else :
                    if word_neg_score > max_neg_sent_word[3] :
                        max_neg_sent_word[0] = idx
                        max_neg_sent_word[1] = word
                        max_neg_sent_word[2] = word_pos_score
                        max_neg_sent_word[3] = word_neg_score
        if max_pos_sent_word[2] >= max_neg_sent_word[3] : 
            words.append(max_pos_sent_word)
        else :
            words.append(max_neg_sent_word)
        return words


    def extract_aspects(self, parsed_sent,  words) :
        """ Extract aspects from a sentence
            Currently, we assume that there is only one aspect in a sentence.
            Input: 
                parsed_sent -- parsed sentence as 
                     [{'address': i, 'deps': [j], 'rel': 'DEP', 'tag': 'POS', 'word': WORD}, ... ]
                words -- a list of sentiment words as [[index, 'word', pos_score, neg_score], ... ]
            return list of aspects [(index, 'word', pos_score, neg_score), ... ]
            For the first version of this function. It just search for the closed noun word as
                the aspect.
                The next version, it will search for the aspect based on the dependency relation
                in the parsed_sent.
        """
        index, word, pos_score, neg_score =  words[0]
        index += 1 # adjust index for parsed sentence list 
        aspects = []
        word_noun = self.get_noun(index, parsed_sent)
        if word_noun is  None  :
            pass
        else :
            aspect = (word_noun[0], word_noun[1], pos_score, neg_score)
            aspects.append(aspect)
        return aspects

    def is_noun(self, w):
        if w is None :
            return False
        else :
            return w in self._POS_NOUNS

    def get_noun(self, index, parsed_sent) :
        """
            find the nearest noun to the indexed word .
            return (index, word)
        """
        node_list = parsed_sent.nodelist
        length = len(node_list)
        l_count = index
        r_count = index
        count = length
        while count > 0 :
            #check left words
            l_count -= 1
            r_count += 1
            if l_count >= 0 :
                l_word = node_list[l_count]               
                w_pos = l_word['tag']
                if self.is_noun(w_pos) == True :
                    return (l_count, l_word['word'])
                else:
                    if r_count < length :
                        r_word = node_list[r_count]               
                        w_pos = r_word['tag']
                        if self.is_noun(w_pos) == True :
                            return (r_count, r_word['word'])
                        else :
                            count -= 1
                    else:
                        pass
            else :
                if r_count < length :
                        r_word = node_list[r_count]               
                        w_pos = r_word['tag']
                        if self.is_noun(w_pos) == True :
                            return (r_count, r_word['word'])
                        else :
                            count -= 1
                else :
                    return None

