#!/usr/bin/env python3

# Python module import
import os
import numpy as np
import math as ma
import argparse
import sys
import csv

from PIL import Image

# keras import
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import RMSprop
from keras.layers import Dense, Activation, LSTM
from keras.preprocessing import sequence

# lstm
import lstm



def lstm_re_train():
	
	# Parse the command line options.
	lstm_path, classes, training_path, training_list, dataset_pickle, label_pickle, create_confusion_matrix = parseOpts( sys.argv )

	model = lstm.lstm_load(lstm_path)
	
	score, acc, confusion_matrix = lstm.lstm_validate(model, classes, evaluation_directory=training_path, training_list=training_list, dataset_pickle_file=dataset_pickle, label_pickle_file=label_pickle, create_confusion_matrix=create_confusion_matrix )

	if create_confusion_matrix is True:
		file = open("confusion_matrix.conf_matrix", "wt")
		writer = csv.writer(file)
		writer.writerows(confusion_matrix)

		print("confusion Martix here")
		print(confusion_matrix)

		# bonus create Bitmap image of results
		img = Image.new('RGB',(len(confusion_matrix) * 10,len(confusion_matrix) * 10),"black")
		pixels = img.load()

		for i in range(img.size[0]):
			for j in range (img.size[1]):
				pixels[i,j] = (0,int(confusion_matrix[int(j/10),int(i/10)] * 255),0)

		img.save('confusion_matrix.bmp')

	
# ----------------------------------------------------------------------------------------------------------------------------------------------------------

# Parse the command line arguments
def parseOpts( argv ):

	# generate parser object
	parser = argparse.ArgumentParser()
	# add arguments to the parser so he can parse the shit out of the command line
	parser.add_argument("-p", "--path", action='store', dest="lstm_path", help="The PATH where the lstm-model will be saved.")
	parser.add_argument("-c", "--classes", action='store', dest="lstm_classes", help="The number of output classes.")
	parser.add_argument("-tp", "--training_path", action='store', dest="training_path", help="The path of the evaluation directory.")
	parser.add_argument("-tl", "--training_list", action='store', dest='training_list', help="A list of training feature in the form: -aL S001,S002,S003,... (overrites -ep)")
	parser.add_argument("-dp", "--dataset_pickle", action='store', dest="dataset_pickle", help="The path to the dataset pickle object. (requires -lp)")
	parser.add_argument("-lp", "--label_pickle", action='store', dest="label_pickle", help="The path to the labels pickle object. (requires -dp)")
	parser.add_argument("-cm", "--confusion_matrix", action='store_true', dest="confusion_matrix", help="set and confusion Matrix will be created")
	

	# finally parse the command line 
	args = parser.parse_args()

	if args.lstm_path:
		lstm_path = args.lstm_path
	else:
		lstm_path = None

	if args.lstm_classes:
		lstm_classes = int(args.lstm_classes)
	else:
		lstm_classes = 1
	
	if args.training_path:
		training_path = args.training_path
	else:
		training_path = None
		
	if args.training_list:
		training_list = args.training_list.split(",")
	else:
		training_list = None

	if args.dataset_pickle:
		dataset_pickle = args.dataset_pickle
	else:
		dataset_pickle = ""
	
	if args.label_pickle:
		label_pickle = args.label_pickle
	else:
		label_pickle = ""

	print ("\nConfiguration:")
	print ("-----------------------------------------------------------------")
	print ("Output classes     : ", lstm_classes)
	print ("Lstm destination   : ", lstm_path)

	return lstm_path, lstm_classes, training_path, training_list, dataset_pickle, label_pickle, args.confusion_matrix

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
	lstm_re_train()