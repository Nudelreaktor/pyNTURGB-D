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

# file dialog
import tkinter as tk
from tkinter import filedialog



def lstm_init(save = False):


	# Parse the command line options.
	save, lstm_path, epochs, classes, hoj_height, training_path, evaluation_path, training_list = parseOpts( sys.argv )

	print("creating neural network...")
	# create neural network
	# 2 Layer LSTM
	model = Sequential()

	# LSTM Schichten hinzufuegen
	model.add(LSTM(hoj_height, input_shape=(None,hoj_height), return_sequences=True))
	model.add(LSTM(hoj_height))	# sehr wichtig

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

	model = lstm_train(model, classes, epochs=epochs, training_directory=training_path, training_list=training_list)
	
	if training_list is None:
		evaluation_path = training_path
	score = lstm_validate(model, classes, evaluation_directory=evaluation_path, training_list=training_list)


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
def lstm_train(lstm_model, classes, epochs=100, training_directory="lstm_train/", training_list=None):
	
	print("train neural network...")
	directories = os.listdir(training_directory)
	
	# Trainingsepochen
	for x in range(0,epochs):
		print("Epoch", x+1, "/", epochs)
		# lade und tainiere jeden HoJ-Ordner im Trainingsverzeichnis
		for directory in directories:
			if to_train(training_list, os.path.basename(directory)):
				training_data = []
				training_labels = []
				hoj_set, labels = get_hoj_data(training_directory + directory, classes)
				training_data.append(hoj_set)
				training_labels.append(labels)
				
				# train neural network
				lstm_model.fit(np.array(training_data), np.array(training_labels), epochs=1, batch_size=1, verbose=0) # epochen 1, weil außerhald abgehandelt; batch_size 1, weil data_sets unterschiedliche anzahl an Frames
			
	return lstm_model

#use this funktion to train the neural network
def lstm_validate(lstm_model, classes, evaluation_directory="lstm_train/", training_list=None):
	
	print("evaluate neural network...")
	directories = os.listdir(evaluation_directory)
	
	accuracy = 0
	n = 0

		# lade und validiere jeden HoJ-Ordner im Validierungsverzeichnis
	for directory in directories:
		if to_evaluate(training_list, os.path.basename(directory)):
			validation_data = []
			validation_labels = []
			data, labels = get_hoj_data(evaluation_directory + directory, classes)
			validation_data.append(data)
			validation_labels.append(labels)
		
			# evaluate neural network
			score, acc = lstm_model.evaluate(np.array(validation_data), np.array(validation_labels), batch_size=1, verbose=0) # batch_size willkuerlich
			accuracy = accuracy + acc
			n += 1
		
	print("Accuracy",accuracy/n)
	return score


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
		label[idx] = 1

	return hoj_set, label


		
# use this funktion to evaluate data in the neural network
def lstm_predict(lstm_model, hoj3d_set):
	prediction = lstm_model.predict(hoj3d_set,batch_size = 1)
	idx = nu.argmax(prediction)[0]
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

	return (not args.test_network), lstm_path, lstm_epochs, lstm_classes, lstm_size, training_path, evaluation_path, training_list

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
	lstm_init(True)