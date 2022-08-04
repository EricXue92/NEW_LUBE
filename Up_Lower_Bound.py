import numpy as np 
import pandas as pd  
import os 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import Dense, Input, Concatenate
from keras.models import Model
from tensorflow.keras import layers
from UpperLower_Control import UpperLower_Control 

class UpperLowerBound:

	def __init__(self, dataset, target):
		self.filepath = 'dataset/'
		self.dataset = dataset
		self.target = target 
		self._load_data()
		self.model = self.build_model()

	# Custom a training model 
	def build_model(self):

		inputs = Input(shape=self.X_train.shape[1:])
		curr = layers.Dense(64, activation='relu', kernel_initializer='normal')(inputs) 
		curr = layers.Dense(32, activation='relu', kernel_initializer='normal')(curr)

		# Lower bound  (head)
		low_bound = layers.Dense(8)(curr) 
		low_bound = layers.Dense(1, name = 'upper_bound')(low_bound) 

		# Upper bound  (head)
		up_bound = layers.Dense(8)(curr)    
		up_bound = layers.Dense(1, name = "lower_bound")(up_bound)

		# Selective (head)
		selective = layers.Dense(16, activation='relu', kernel_initializer='normal')(curr)
		selective = Dense(1, activation='sigmoid')(selective)

		selective_outputs = Concatenate(axis=1, name="combined_output") ( [low_bound, up_bound, selective] )
		

		# auxiliary outputs
		low_bound = layers.Dense(8)(curr) 
		low_bound = layers.Dense(1, name = 'aux_upper_bound')(low_bound) 

		# Upper bound  (head)
		up_bound = layers.Dense(8)(curr)    
		up_bound = layers.Dense(1, name = "aux_lower_bound")(up_bound)

		auxiliary_outputs = Concatenate(axis=1, name="auxiliary_output") ([low_bound, up_bound])

		return UpperLower_Control(inputs, selective_outputs)

	# y is the target value 
	def _load_data(self):
		file_path = os.path.join(self.filepath, self.dataset)

		if file_path.split('.')[-1] == 'xls' or file_path.split('.')[-1] == 'xlsx' :
			df = pd.read_excel(file_path)
		else:
		 	df = pd.read_csv(file_path)

		X = df.drop(self.target, axis = 1)
		y = df[self.target].values.reshape(-1,1)

		# Save the range of y befor transformation
		self.range = max(y) - min(y)

		# Scale data to [0,1]
		X, y = self.scaled_data(X, y)

		# Standardization to x ~ N(0, 1) 
		#X, y = self.standarlize_data(X,y)

        # Split data
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # random_state=42

		X_train = X_train.astype('float32')
		X_test = X_test.astype('float32')
		y_train = y_train.astype('float32')
		y_test = y_test.astype('float32')

		# Change y based on the model 
		self.y_train = np.repeat(y_train, [2], axis = 1)
		self.y_test = np.repeat(y_test, [2], axis = 1)
		self.y_train = np.hstack((self.y_train, np.zeros((self.y_train.shape[0], 1), dtype=self.y_train.dtype)))
		self.y_test = np.hstack((self.y_test, np.zeros((self.y_test.shape[0], 1), dtype=self.y_test.dtype)))
		self.X_train, self.X_test = X_train,  X_test

		# Scale data to [0,1]
	def scaled_data(self, X, y):

		self.scaler_X = MinMaxScaler()
		self.scaler_y = MinMaxScaler()

		#inverse_transform()
		X = self.scaler_X.fit_transform(X)
		y = self.scaler_y.fit_transform(y)
		return X, y 

        # Transfor scaled data to original data
	def reversed_data(self, y):
		return self.scaler_y.inverse_transform(y)

		# Standardization to N(0, 1) 
	def standarlize_data(self, X,y):
		X = StandardScaler().fit_transform(X)
		y = StandardScaler().fit_transform(y)
		return X, y

	def predict(self, x=None, batch_size=256):
		if x is None:
			x = self.X_test
		return self.model.predict(x, batch_size)
