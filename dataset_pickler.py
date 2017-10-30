#!/usr/bin/env python3

# Python module import
import os
import numpy as np
import math as ma
import argparse
import sys
import pickle
import time
import datetime


def dataset_pickler():

	# Parse the command line arguments
	training_directory, training_list, dataset_path, labels_path, classes, validation_dataset_path, validation_labels_path = parseOpts( sys.argv )

	training_data = []
	training_labels = []
	evaluation_data = []
	evaluation_labels = []
	idx = 0

	# pickeling
	if(dataset_path is None or labels_path is None or validation_dataset_path is None or validation_labels_path is None):

		#create timestamp for filenames
		timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')

		dataset_path = "dataset_" + timestamp
		labels_path = "labels_" + timestamp
		validation_dataset_path = "dataset_v_" + timestamp
		validation_labels_path = "labels_v_" + timestamp
		print("new dataset file", dataset_path)

	directories = os.listdir(training_directory)
	directories_len = len(directories)
	
	for directory in directories:
		if to_train(training_list, os.path.basename(directory)):
			hoj_set, labels = get_hoj_data(training_directory + directory, classes)
			training_data.append(hoj_set)
			training_labels.append(labels)
		else:
			hoj_set, labels = get_hoj_data(training_directory + directory, classes)
			evaluation_data.append(hoj_set)
			evaluation_labels.append(labels)

		idx = idx+1
		print("Loading ... ", idx, "/", directories_len, end="\r")


	print("write pickles")
	dataset_file = open(dataset_path,"wb")
	pickle.dump(training_data,dataset_file,protocol=pickle.HIGHEST_PROTOCOL)
	dataset_file.close()

	labels_file = open(labels_path,"wb")
	pickle.dump(training_labels,labels_file,protocol=pickle.HIGHEST_PROTOCOL)
	labels_file.close()

	validation_dataset_file = open(validation_dataset_path,"wb")
	pickle.dump(evaluation_data,validation_dataset_file,protocol=pickle.HIGHEST_PROTOCOL)
	validation_dataset_file.close()

	validation_labels_file = open(validation_labels_path,"wb")
	pickle.dump(evaluation_labels,validation_labels_file,protocol=pickle.HIGHEST_PROTOCOL)
	validation_labels_file.close()
	print("finnished")




def get_hoj_data(directory, classes, pickle_file=""):
	hoj_set_files = os.listdir(directory)
	data = []
	hoj_set = []
	label = np.zeros(classes)
	# alle dateien laden, in einer Matrix peichern
	for hoj_file in hoj_set_files:
		file = open(directory + "/" + hoj_file,'rb')
		hoj_array = np.load(file)
		file.close()

		hoj_set.append(hoj_array)

	# lade Labels (test output)
	idx = int(directory[-3:])
	label[idx - 1] = 1


	return np.array(hoj_set), label

# A small function to skip actions which are not in the training_list
def to_train( training_list, _skeleton_filename_ ):
	# If an training_list is given 
	if( training_list is not None ):
		for key in training_list:
			if( key in _skeleton_filename_ ):
				# If the action of the skeleton file is in the training_list.
				return True
	# If no training_list is given
	else:
		return True

	# If the action of the skeleton file is not in the training_list.
	return False

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

# Parse the command line arguments
def parseOpts( argv ):

	# generate parser object
	parser = argparse.ArgumentParser()
	# add arguments to the parser so he can parse the shit out of the command line
	parser.add_argument("-p", "--path", action='store', dest="directory", help="The path of the existing dataset.")
	parser.add_argument("-d", "--dataset_path", action='store', dest="dataset_path", help="The path of the saved dataset.")
	parser.add_argument("-l", "--labels_path", action='store', dest="labels_path", help="The path of the saved labels.")
	parser.add_argument("-vd", "--validation_dataset_path", action='store', dest="validation_dataset_path", help="The path of the saved labels.")
	parser.add_argument("-vl", "--validation_labels_path", action='store', dest="validation_labels_path", help="The path of the saved labels.")

	parser.add_argument("-tl", "--training_list", action='store', dest='training_list', help="A list of training feature in the form: -tl S001,S002,S003,... (overrites -ep)")
	parser.add_argument("-c", "--classes", action='store', dest="lstm_classes", help="The number of output classes.")	

	# finally parse the command line 
	args = parser.parse_args()

	
	if args.directory:
		directory = args.directory
	else:
		directory = None
		
	if args.training_list:
		training_list = args.training_list.split(",")
	else:
		training_list = None

	if args.dataset_path:
		dataset_path = args.dataset_path
	else:
		dataset_path = None

	if args.labels_path:
		labels_path = args.labels_path
	else:
		labels_path = None

	if args.validation_dataset_path:
		validation_dataset_path = args.validation_dataset_path
	else:
		validation_dataset_path = None

	if args.validation_labels_path:
		validation_labels_path = args.validation_labels_path
	else:
		validation_labels_path = None

	if args.lstm_classes:
		lstm_classes = int(args.lstm_classes)
	else:
		lstm_classes = 2


	print ("\nConfiguration:")
	print ("-----------------------------------------------------------------")

	return directory, training_list, dataset_path, labels_path, lstm_classes, validation_dataset_path, validation_labels_path

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
	dataset_pickler()