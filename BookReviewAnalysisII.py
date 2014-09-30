import sys
import os
from Bookreview import *
from os.path import isfile, join

FILE_PATH = 'g:\\projects\\SentimentAnalysis\\Test'
FILE_NAME = 'book_review.txt'
RESULTS = 'review_results.txt'
print('Running Book Review Analysis!')
input_file = join(FILE_PATH, FILE_NAME)
output_file = join(FILE_PATH, RESULTS)
book_review = Bookreview()
res = (0, 0.0)
res, sents = book_review.do_analysis(input_file)
output_obj = open(output_file, 'w')
output_obj.write('document ' + FILE_NAME + ' \n')
output_obj.write(str(res[0])+', ' + str(res[1]) + '\n')
output_obj.close()
print('All Done!')