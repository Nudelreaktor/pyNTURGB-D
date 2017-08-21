#!/usr/bin/env python3

# Python module import
import os
import numpy as np
import math as ma

# keras import
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import RMSprop
from keras.layers import Dense, Activation, LSTM

# file dialog
import tkinter as tk
from tkinter import filedialog



def lstm_init(save = False):
	hoj_height = 168
	classes = 55
	
	##############################################
	# Training Data                              #
	##############################################
	print("reading test data...")
	directories = os.listdir("lstm_train")
	training_data = []
	training_labels = []
	for directory in directories:
		hoj_set_files = os.listdir("lstm_train/" + directory)
		hoj_set = []
		for hoj_file in hoj_set_files:
			# alle laden, in einer Matrix peichern
			file = open("lstm_train/" + directory + "/" + hoj_file,'rb')
			hoj_array = np.load(file)
			file.close()

			hoj_set.append(hoj_array)
			
		# lade Labels (test output)
		label_index = int(directory[-3:])
		label = np.zeros(classes)
		label[label_index] = 1
		
		training_data.append(np.array(hoj_set))
		training_labels.append(label)
	
	print(np.array(training_data).shape)
	print(training_data)
	
	
	##############################################
	# Validation Data                            #
	##############################################
	print("reading validation data...")
	directories = os.listdir("lstm_validate")
	validation_data = []
	validation_labels = []
	for directory in directories:
		hoj_set_files = os.listdir("lstm_validate/" + directory)
		hoj_set = []
		for hoj_file in hoj_set_files:
			# alle laden, in einer Matrix peichern
			file = open("lstm_validate/" + directory + "/" + hoj_file,'rb')
			hoj_array = np.load(file)
			file.close()

			hoj_set.append(hoj_array)
			
		# lade Labels (test output)
		label_index = int(directory[-3:])
		label = np.zeros(classes)
		label[label_index] = 1
		
		validation_data.append(np.array(hoj_set))
		validation_labels.append(label)
		
	print(np.array(validation_data).shape)
		
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
	
	print("train neural network...")
	# train neural network
	#
	#
	#
	# TODO FIT_GENERATOR
	#
	#
	#
	#
	#
	model.fit(training_data, training_labels, epochs=100, batch_size=32, verbose=2) # epochen willkuerlich; batch_size willkuerlich
	score = model.evaluate(validation_data, validation_labels, batch_size=32) # batch_size willkuerlich

	print("neural Network score: " + score)
	
	
	if save == True:
		# save neural network
		# Open a save dialog
		f = filedialog.asksaveasfilename(initialdir=store_path, title="store model", filetypes=(("Model files","*.h5"),("all files","*.*")))
		if f is not None:
			model.save(f)

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

		
		
# use this funktion to evaluate data in the neural network
def lstm_predict(lstm_model, hoj3d_set):
	prediction = lstm_model.predict(hoj3d_set,batch_size = 1)
	idx = nu.argmax(prediction)[0]
	return idx,prediction[idx],prediction

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
	lstm_init(True)