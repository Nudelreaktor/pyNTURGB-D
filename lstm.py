#!/usr/bin/env python3

# Python module import
import os
import numpy as np
import math as ma
import argparse

# keras import
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import RMSprop
from keras.layers import Dense, Activation, LSTM

# file dialog
import tkinter as tk
from tkinter import filedialog



def lstm_init(save = False):


	# Parse the command line options.
	save, lstm_path = parseOpts( sys.argv )

	hoj_height = 168
	classes = 61

	print("creating neural network...")
	# create neural network
	# 2 Layer LSTM
	model = Sequential()

	# LSTM Schichten hinzufuegen
	model.add(LSTM(hoj_height, input_shape=(None,hoj_height), return_sequences=True))
	#model.add(LSTM(hoj_height))	# vielleicht optional

	# voll vernetzte Schicht zum Herunterbrechen vorheriger Ausgabedaten auf die Menge der Klassen 
	model.add(Dense(classes))
	
	# Aktivierungsfunktion = Transferfunktion
	# Softmax -> hoechsten Wert hervorheben und ausgaben normalisieren
	model.add(Activation('softmax'))
	
	# lr = Learning rate
	# zur "Abkuehlung" des Netzwerkes
	optimizer = RMSprop(lr=0.01)
	# categorical_crossentropy -> ein Ausgang 1 der Rest 0
	model.compile(loss='categorical_crossentropy',optimizer=optimizer)

	model = lstm_train(model, epochs=100, classes=classes)
	score = lstm_validate(model, classes=classes)
	
	

	# print("neural Network score: " + score)


	print("network creation succesful! \\(^o^)/")
	
	
	if save is True:
		if lstm_path is not None:
			model.save(lstm_path)
		else:
			# save neural network
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
def lstm_train(lstm_model, epochs=100, classes=61):
	
	print("train neural network...")
	directories = os.listdir("lstm_train/")
	for x in range(0,epochs):
		print("Epoch: ", x+1, "/", epochs)
		for directory in directories:
			training_data, training_labels = get_hoj_data("lstm_train/" + directory, classes=classes)
			
			# train neural network
			history = lstm_model.fit(np.array(training_data), np.array(training_labels), epochs=1, batch_size=1, verbose=0) # epochen willkuerlich; batch_size willkuerlich
			print(history.history)
	return lstm_model

#use this funktion to train the neural network
def lstm_validate(lstm_model, classes=61):
	
	print("evaluate neural network...")
	directories = os.listdir("lstm_validate/")
	for directory in directories:
		validation_data, validation_labels = get_hoj_data("lstm_validate/" + directory, classes=classes)
		
		# evaluate neural network
		score = lstm_model.evaluate(np.array(validation_data), np.array(validation_labels), batch_size=32) # batch_size willkuerlich
		print(score)
	return score



def get_hoj_data(directory, classes=61):
	hoj_set_files = os.listdir(directory)
	data = []
	hoj_set = []
	labels = []
	for hoj_file in hoj_set_files:
		# alle laden, in einer Matrix peichern
		file = open(directory + "/" + hoj_file,'rb')
		hoj_array = np.load(file)
		file.close()

		hoj_set.append(hoj_array)
		
		# lade Labels (test output)
		idx = int(directory[-3:])

		label = np.zeros(classes)
		label[idx] = 1
		labels.append(label)

	data.append(hoj_set)


	all_labels = []
	all_labels.append(labels)
	return data, all_labels


		
# use this funktion to evaluate data in the neural network
def lstm_predict(lstm_model, hoj3d_set):
	prediction = lstm_model.predict(hoj3d_set,batch_size = 1)
	idx = nu.argmax(prediction)[0]
	return idx,prediction[idx],prediction

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

# Parse the command line arguments
def parseOpts( argv ):

	skeleton_name = ""

	# generate parser object
	parser = argparse.ArgumentParser()
	# add arguments to the parser so he can parse the shit out of the command line
	parser.add_argument("-t", "--test", action='store_true', dest='test_network', help="if set the created neural network won't be saved. (overites -p)")
	parser.add_argument("-p", "--path", action='store', dest="lstm_path", help="The PATH where the lstm-model will be saved.")

	# finally parse the command line 
	args = parser.parse_args()

	if args.lstm_path:
		lstm_path = args.lstm_path
	else:
		lstm_path = None

	print ("\nConfiguration:")
	print ("----------------------------------------------------------------------------------------------------------------------------")
	print ("Lstm destination   : ", args.lstm_path)
	if args.test_network is True:
		print("Network won't be saved!")
	else:
		print("Network will be saved")

	return args.test_network, lstm_path

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
	lstm_init(True)