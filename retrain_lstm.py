#!/usr/bin/env python3

# Python module import
import os
import numpy as np
import math as ma
import argparse
import sys

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
	lstm_path, epochs, classes, training_path, training_list = parseOpts( sys.argv )

	model = lstm.lstm_load(lstm_path)
	
	model = lstm.lstm_train(model, classes, epochs=epochs, training_directory=training_path, training_list=training_list)
	
	model.save(lstm_path)

	
# ----------------------------------------------------------------------------------------------------------------------------------------------------------

# Parse the command line arguments
def parseOpts( argv ):

	# generate parser object
	parser = argparse.ArgumentParser()
	# add arguments to the parser so he can parse the shit out of the command line
	parser.add_argument("-p", "--path", action='store', dest="lstm_path", help="The PATH where the lstm-model will be saved.")
	parser.add_argument("-e", "--epochs", action='store', dest="lstm_epochs", help="The number of training epochs.")
	parser.add_argument("-c", "--classes", action='store', dest="lstm_classes", help="The number of output classes.")
	parser.add_argument("-tp", "--training_path", action='store', dest="training_path", help="The path of the training directory.")
	parser.add_argument("-tl", "--training_list", action='store', dest='training_list', help="A list of training feature in the form: -aL S001,S002,S003,... (overrites -ep)")
	

	# finally parse the command line 
	args = parser.parse_args()

	if args.lstm_path:
		lstm_path = args.lstm_path
	else:
		lstm_path = None

	if args.lstm_epochs:
		lstm_epochs = int(args.lstm_epochs)
	else:
		lstm_epochs = 10

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

	print ("\nConfiguration:")
	print ("-----------------------------------------------------------------")
	print ("Output classes     : ", lstm_classes)
	print ("Training Epochs    : ", lstm_epochs)
	print ("Lstm destination   : ", lstm_path)

	return lstm_path, lstm_epochs, lstm_classes, training_path, training_list

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
	lstm_re_train()