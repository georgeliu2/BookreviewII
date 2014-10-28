import sys
import os
from Bookreview import *
from os.path import isfile, join
import Aspect

FILE_PATH = ""      #input  file path
FILE_NAME = ""      #inut file name
RESULTS = ""        #output file name
SWN_FILENAME = ""   #SentiWordNet   dataset file
WORKING_DIR = ""    #Malt Parser working dir
MCO_FILE = ""       #Trained model

if __name__ == "__main__" :
    config_file = sys.argv[1]            #arg 1 is the config file
    #get config parameters from config_file
    if os.path.isfile(config_file) == False :
        print "Cannot find the config file: ", config_file
        exit()
    
    with open(config_file) as f:
        content = f.readlines()
        line = content[0]  
        FILE_PATH = line.split()[0]
        line = content[1]  
        FILE_NAME = line.split()[0]
        line = content[2]  
        RESULTS = line.split()[0]
        line = content[3]  
        SWN_FILENAME = line.split()[0]
        line = content[4]  
        WORKING_DIR = line.split()[0]
        line = content[5]  
        MCO_FILE = line.split()[0]
        f.close()

    input_file = join(FILE_PATH, FILE_NAME)
    output_file = join(FILE_PATH, RESULTS)
    book_review = Bookreview(SWN_FILENAME, WORKING_DIR, MCO_FILE)
    res = (0, 0.0)
    res, aspects = book_review.do_analysis(input_file)
    output_obj = open(output_file, 'w')
    output_obj.write('document ' + FILE_NAME + ' \n')
    output_obj.write(str(res[0])+', ' + str(res[1]) + '\n')
    output_obj.write("Aspects: \n")
    for key, value in aspects.iteritems():    
        output_obj.write("Name: "+str(key) + "; frequency: " + str(value[0]) + "; pos score " + str(value[1]) + "; neg score " +str(value[2])+ " \n")
    output_obj.close()
    print('All Done!')