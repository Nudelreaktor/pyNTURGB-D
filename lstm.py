#!/usr/bin/env python3

# Python module import
import os
import numpy as np
import math as ma
import argparse
import sys
import random
import time
import datetime
import json
import pickle

# keras import
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import RMSprop
from keras.layers import Dense, Activation, LSTM
from keras.preprocessing import sequence

# confusion matrix
from sklearn.metrics import confusion_matrix

# file dialog
import tkinter as tk
from tkinter import filedialog



def lstm_init(save = False):


	# Parse the command line options.
	save, lstm_path, epochs, classes, hoj_height, training_path, evaluation_path, training_list, layer_sizes, dataset_pickle_path, label_pickle_path, sample_strategy = parseOpts( sys.argv )

	print("creating neural network...")
	# create neural network
	# 2 Layer LSTM
	model = Sequential()

	# LSTM Schichten hinzufuegen
	if(len(layer_sizes) == 1):
		model.add(LSTM(int(layer_sizes[0]), input_shape=(None,hoj_height)))
	else:
		for i in range(len(layer_sizes)):
			if i == 0:
				model.add(LSTM(int(layer_sizes[i]), input_shape=(None,hoj_height), return_sequences=True))
			else:
				if i == len(layer_sizes) - 1:
					model.add(LSTM(int(layer_sizes[i])))	# sehr wichtig
				else:
					model.add(LSTM(int(layer_sizes[i]), return_sequences=True))


	# voll vernetzte Schicht zum Herunterbrechen vorheriger Ausgabedaten auf die Menge der Klassen 
	model.add(Dense(classes))
	
	# Aktivierungsfunktion = Transferfunktion
	# Softmax -> hoechsten Wert hervorheben und ausgaben normalisieren
	model.add(Activation('softmax'))
	
	# lr = Learning rate
	# zur "Abkuehlung" des Netzwerkes
	optimizer = RMSprop(lr=0.001)
	# categorical_crossentropy -> ein Ausgang 1 der Rest 0
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	model.summary()

	model = lstm_train(model, classes, epochs=epochs, training_directory=training_path, training_list=training_list, dataset_pickle_file=dataset_pickle_path, label_pickle_file=label_pickle_path, _sample_strategy=sample_strategy)
	
	#if training_list is not None:
	#	evaluation_path = training_path
	#score = lstm_validate(model, classes, evaluation_directory=evaluation_path, training_list=training_list)


	print("network creation succesful! \\(^o^)/")
	
	
	# save neural network
	if save is True:
		if lstm_path is not None:
			model.save(lstm_path)
		else:
			# Open a save dialog
			f = filedialog.asksaveasfilename(title="store model", filetypes=(("Model files","*.h5"),("all files","*.*")))
			if f is not None and f is not "":
				model.save(f + ".h5")

	return model
	
	
# use this funktion to load a trained neural network
def lstm_load(filename = None):
	if filename is not None:
		return load_model(filename)

	f = filedialog.askopenfilename(filetypes=(("Model files","*.h5"),("all files","*.*")))
	if f is None or f is "":
		return None
	else:
		return load_model(f)

#use this funktion to train the neural network
def lstm_train(lstm_model, classes, epochs=100, training_directory="lstm_train/", training_list=None, dataset_pickle_file="", label_pickle_file="", _sample_strategy="random"):
	
	print("train neural network...")
	directories = os.listdir(training_directory)
	directories_len = len(directories)

	complete_hoj_data = None

	#create timestamp for filenames
	timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')

	# create a history file
	history_file_name = "history_" + timestamp + ".json"
	history_file = open(history_file_name,"wt")


	dataset_pickle_object = None
	labels_pickle_object = None
	if(os.path.isfile(dataset_pickle_file) and os.path.isfile(label_pickle_file)):
		dataset_file = open(dataset_pickle_file,"rb")
		dataset_pickle_object = pickle.load(dataset_file)
		dataset_file.close()

		label_file = open(label_pickle_file,"rb")
		labels_pickle_object = pickle.load(label_file)
		label_file.close()

	# Trainingsepochen
	for x in range(0,epochs):
		print("Epoch", x+1, "/", epochs)
		# lade und tainiere jeden HoJ-Ordner im Trainingsverzeichnis
		training_data = []
		training_labels = []
		idx = 0

		# read dataset and labels
		
		if(os.path.isfile(dataset_pickle_file) and os.path.isfile(label_pickle_file)):
			# eight buckets
			for _set in dataset_pickle_object:

				selected_set = get_eight_buckets(_set, _sample_strategy)


				training_data.append(selected_set)

			training_labels = labels_pickle_object


		else:
			for directory in directories:
				if to_train(training_list, os.path.basename(directory)):
					hoj_set, labels = get_hoj_data(training_directory + directory, classes)
					training_data.append(hoj_set)
					training_labels.append(labels)
				idx = idx+1
				print("Loading ... ", idx, "/", directories_len, end="\r")

		# train neural network
		training_history = lstm_model.fit(np.array(training_data), np.array(training_labels), epochs=1, batch_size=32, verbose=1) # epochen 1, weil außerhald abgehandelt; batch_size 1, weil data_sets unterschiedliche anzahl an Frames
		json.dump(training_history.history,history_file)
		history_file.write("\n")
	
	history_file.close()
			
	return lstm_model

#use this funktion to train the neural network
def lstm_validate(lstm_model, classes, evaluation_directory="lstm_train/", training_list=None, dataset_pickle_file="", label_pickle_file="", create_confusion_matrix=False, _sample_strategy="random"):
	
	print("evaluate neural network...")
	directories = os.listdir(evaluation_directory)
	directories_len = len(directories)
	validation_data = []
	validation_labels = []
	
	accuracy = 0
	n = 0
	idx = 0

	# read dataset and labels
	
	if(os.path.isfile(dataset_pickle_file) and os.path.isfile(label_pickle_file)):

		dataset_file = open(dataset_pickle_file,"rb")
		dataset_pickle_object = pickle.load(dataset_file)
		dataset_file.close()

		label_file = open(label_pickle_file,"rb")
		labels_pickle_object = pickle.load(label_file)
		label_file.close()

		# eight buckets
		for _set in dataset_pickle_object:

			selected_set = get_eight_buckets(_set, _sample_strategy)

			validation_data.append(selected_set)

		validation_labels = labels_pickle_object

	else:
		# lade und validiere jeden HoJ-Ordner im Validierungsverzeichnis
		for directory in directories:
			if to_evaluate(training_list, os.path.basename(directory)):
				data, labels = get_hoj_data(evaluation_directory + directory, classes)
				validation_data.append(data)
				validation_labels.append(labels)
			
						
			idx = idx+1
			print(idx, "/", directories_len, end="\r")


	# evaluate neural network
	score, acc = lstm_model.evaluate(np.array(validation_data), np.array(validation_labels), batch_size=32, verbose=0) # batch_size willkuerlich
			
	print("Accuracy:",acc)

	if create_confusion_matrix is True:
		predictions = lstm_model.predict(np.array(validation_data),batch_size = 32)
		
		predicted_labels = []
		real_labels = []

		for k in range(len(predictions)):
			predicted_idx = np.argmax(predictions[k])

			label_idx = np.argmax(validation_labels[k])
			
			real_labels.append(label_idx)
			predicted_labels.append(predicted_idx)


		cnf_matrix = confusion_matrix(real_labels, predicted_labels)
		cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
		return score, acc, cnf_matrix


	return score, acc, None


def get_hoj_data(directory, classes):
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

	selected_hoj_set = get_eight_buckets(hoj_set)

	return np.array(selected_hoj_set), label


		
# use this funktion to evaluate data in the neural network
def lstm_predict(lstm_model, hoj3d_set):
	prediction = lstm_model.predict(hoj3d_set,batch_size = 1)
	idx = np.argmax(prediction)[0]
	return idx,prediction[0][0][idx],prediction
	
# ----------------------------------------------------------------------------------------------------------------------------------------------------------

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

# A small function to skip actions which are in the training_list
def to_evaluate( training_list, _skeleton_filename_ ):
	# If an training_list is given 
	if( training_list is not None ):
		for key in training_list:
			if( key in _skeleton_filename_ ):
				# If the action of the skeleton file is in the training_list.
				return False

	# If the action of the skeleton file is not in the training_list.
	return True


def get_eight_buckets( hoj_set, _sample_strategy="random" ):

	# Get some informations about the data
	number_of_frames = len(hoj_set)
	_number_of_subframes = 8
	frame = []

	# Compute the size of the 8 buckets depending of the number of frames of the set.
	bucket_size = ma.floor( number_of_frames / _number_of_subframes )
	remain = number_of_frames - ( bucket_size * _number_of_subframes )
	gap = ma.floor(remain / 2.0)

	# Take a random frame from each bucket and store it as array entry in the _svm_structure ( 8 per )
	for k in range(0,_number_of_subframes):

		# Choose the sampling strategy
		# First frame per bucket
		if( _sample_strategy == "first"):
			random_frame_number = int(gap+(k*bucket_size)+1)
		# Mid frame per bucket
		elif( _sample_strategy == "mid"):
			random_frame_number = int(gap+(k*bucket_size)+int(ma.floor(bucket_size/2)))
		# Last frame per bucket
		elif( _sample_strategy == "last"):
			random_frame_number = int(gap+(k*bucket_size)+bucket_size)
		# Random frame per bucket
		else:
			# Get the random frame -> randint(k(BS),k+1(BS)) ==> k-1(B) < randomInt < k(B)
			random_frame_number = random.randint((gap+(k*bucket_size)),(gap+((k+1)*bucket_size)) )

		# Convert the frame to the svm structure 
		# Get the random frame and the corresponding label
		if( random_frame_number > 0 ):
			# Collect the data from the 8 buckets in a list.
			frame.append(hoj_set[random_frame_number-1]);
		else:
			# Collect the data from the 8 buckets in a list.
			frame.append(hoj_set[random_frame_number]);

	return frame


	
# ----------------------------------------------------------------------------------------------------------------------------------------------------------

# Parse the command line arguments
def parseOpts( argv ):

	# generate parser object
	parser = argparse.ArgumentParser()
	# add arguments to the parser so he can parse the shit out of the command line
	parser.add_argument("-t", "--test", action='store_true', dest='test_network', help="if set the created neural network won't be saved. (overites -p)")
	parser.add_argument("-p", "--path", action='store', dest="lstm_path", help="The PATH where the lstm-model will be saved.")
	parser.add_argument("-e", "--epochs", action='store', dest="lstm_epochs", help="The number of training epochs.")
	parser.add_argument("-c", "--classes", action='store', dest="lstm_classes", help="The number of output classes.")
	parser.add_argument("-s", "--input_size", action='store', dest="lstm_size", help="The number of input fields.")
	parser.add_argument("-tp", "--training_path", action='store', dest="training_path", help="The path of the training directory.")
	parser.add_argument("-ep", "--evaluation_path", action='store', dest="evaluation_path", help="The path of the evaluation directory.")
	parser.add_argument("-tl", "--training_list", action='store', dest='training_list', help="A list of training feature in the form: -tl S001,S002,S003,... (overrites -ep)")
	parser.add_argument("-ls", "--layer_sizes", action='store', dest='layer_sizes', help="A list of sizes of the LSTM layers (standart: -ls 16,16)")
	parser.add_argument("-dp", "--dataset_pickle", action='store', dest="dataset_pickle", help="The path to the dataset pickle object. (requires -lp)")
	parser.add_argument("-lp", "--label_pickle", action='store', dest="label_pickle", help="The path to the labels pickle object. (requires -dp)")
	parser.add_argument("-bs", "--bucket_strategy", action='store', dest='bucket_strategy', help="random, first, mid, last")

	

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

	if args.lstm_size:
		lstm_size = int(args.lstm_size)
	else:
		lstm_size = 2
	
	if args.training_path:
		training_path = args.training_path
	else:
		training_path = None
	
	if args.evaluation_path:
		evaluation_path = args.evaluation_path
	else:
		evaluation_path = None
		
	if args.training_list:
		training_list = args.training_list.split(",")
	else:
		training_list = None

	if args.layer_sizes:
		layer_sizes = args.layer_sizes.split(",")
	else:
		layer_sizes = [16,16]

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
	print ("Input size         : ", lstm_size)
	print ("Output classes     : ", lstm_classes)
	print ("Training Epochs    : ", lstm_epochs)
	print ("Lstm destination   : ", lstm_path)
	if args.test_network is True:
		print("Network won't be saved!")
	else:
		print("Network will be saved")

	return (not args.test_network), lstm_path, lstm_epochs, lstm_classes, lstm_size, training_path, evaluation_path, training_list, layer_sizes, dataset_pickle, label_pickle, args.bucket_strategy

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
	lstm_init(True)