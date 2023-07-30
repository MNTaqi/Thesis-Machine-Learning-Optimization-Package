#Artificial Neural Network
PYTHONHASHSEED = 0
#Library Importing 
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as py
from keras.models import Sequential
from keras.layers import Dense
import os
import tensorflow as tf
import random as rn
from keras import backend as K
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
import pickle
import openpyxl
from openpyxl import load_workbook
import joblib
tf.compat.v1.disable_v2_behavior()
standard_scaler=StandardScaler()
normalizer=Normalizer()
min_max_scaler=MinMaxScaler()
#Direction to Project File
os.chdir(r"C:\Users\Taqi\OneDrive - Institut Teknologi Bandung\Desktop\Kuliah\S7\TA\.Phyton\ANN\.TrialNew")

#1. Data Preprocessing Process
#Import Dataset values
dataset = pd.read_excel('ANN_Data_Numeric.xlsx', sheet_name = 'ANN_Data')

#Input Data Input
X = dataset.iloc[0:,1:7].values
#Input Data Output
dfY= dataset.iloc[0:,6:7].values
Y=min_max_scaler.fit_transform(dfY)
DFY = pd.DataFrame(Y)

#Encoding Data Input Categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
trans = ColumnTransformer([('Data',OneHotEncoder(),[0,1,3])],remainder='passthrough')
X_pre = np.array(trans.fit_transform(X), dtype=np.float64)
X_pre = X_pre[:,:]
DFX = pd.DataFrame(X_pre)
#Dummy trap prevention
del DFX[DFX.columns[0]]
del DFX[DFX.columns[5]]
del DFX[DFX.columns[10]]
#hapus data output
del DFX[DFX.columns[15]]
X = np.matrix(DFX) 
X = pd.DataFrame(X)

#2. Initializing ANN Architecture
#Initialize random seed in order to get the reproducible result
np.random.seed(123)
rn.seed(12345)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads = 1, inter_op_parallelism_threads = 1)
tf.random.set_seed(1234)
sess = tf.compat.v1.Session(graph = tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)
n_input = X.shape[1]
#Initialize the sequence of layers
regressor = Sequential()
# Add input layer and first hidden layer to ANN Architecture
regressor.add(Dense(units=20, activation='linear', input_dim=n_input))
# Add another layers in ANN Architecture
regressor.add(Dense(units=10, activation='relu'))
regressor.add(Dense(units=10, activation='sigmoid'))
regressor.add(Dense(units=10, activation='relu'))
regressor.add(Dense(units=10, activation='sigmoid'))
regressor.add(Dense(units=10, activation='relu'))
regressor.add(Dense(units=10, activation='sigmoid'))
regressor.add(Dense(units=10, activation='relu'))
# Add output layer in ANN Arcitecture
regressor.add(Dense(units=1, activation='linear'))

#Compile the ANN model Using Sthocastic Gradient Descent
regressor.compile(optimizer = 'adam',loss = 'mean_squared_error',metrics = ['mse','mae'] )
tf.keras.optimizers.Adam(learning_rate=0.01)
regressor.summary()

#3. Fitting ANN to the Optimization Regression Model
#Fitting to training set
history = regressor.fit(X, Y, steps_per_epoch = 300, validation_steps= 200,validation_split = 0.2, batch_size = 5,nb_epoch = 2000, shuffle = True, verbose = 2)

#crosscheck hasil Y dengan model X
Y_predict = regressor.predict(X)
print(Y_predict)
print(len(DFY))
DFYP = pd.DataFrame(Y_predict)
#DFYPNI = pd.concat([DFY, DFYP, pd.DataFrame(((DFY-DFYP).pow(2))/len(DFY))], ignore_index=True, axis = 1)
#DFYPNI.to_excel('Input vs Predicted Y_Norm.xlsx')
Y_inverse = min_max_scaler.inverse_transform(Y_predict)
DFYP = pd.DataFrame(Y_inverse)
DFYP.to_excel('Predicted Y_EA.xlsx')

#Model Evaluation
from keras.utils.vis_utils import plot_model
from keras.utils.layer_utils import print_summary
tf.keras.utils.plot_model(regressor, to_file='regressor.png')
model = regressor.get_weights()
print_summary(regressor)
DFmodel = pd.DataFrame(model)

#Pull weights and biases to excel
weightexcel = pd.ExcelWriter('Weight Function EA.xlsx', engine='openpyxl')
DFmodel.to_excel(weightexcel, sheet_name="All")
weightexcel.save()
for i in range(len(regressor.layers)):
 book = openpyxl.load_workbook('Weight Function EA.xlsx')
 weightexcel._book = book
 weight = pd.DataFrame(regressor.layers[i].get_weights()[0])
 bias = pd.DataFrame(regressor.layers[i].get_weights()[1])
 weight.to_excel(weightexcel, sheet_name="Weight "+str(i))
 bias.to_excel(weightexcel, sheet_name="Bias "+str(i))
 weightexcel.save()
 print("Wait for Weight Function to be written; "+str(i)+"/"+str(len(regressor.layers)))
weightexcel.close()

#Export ANN model
filename = 'model_EA.h5'
model = regressor.save(filename)

#Save Data Test for Model Evaluation in NSGA -II Optimization
path = r"Data_Test_EA.xlsx"
writer = pd.ExcelWriter(path,engine ="openpyxl")
dfX = pd.DataFrame(X)
dfX.to_excel(writer,sheet_name="X_test")
dfY = pd.DataFrame(DFY)
dfY.to_excel(writer,sheet_name="Y_test")
writer.save()
writer.close()

#Visualization Data
#Plot training & validation accuracy
py.figure(1)
py.plot(history.history['loss'], 'blue')
py.plot(history.history['val_loss'], 'g--')
py.title('Model Loss')
py.ylabel('Loss')
py.xlabel('Epoch')
py.legend(['Train', 'Test'], loc = 'upper right')
py.show()

