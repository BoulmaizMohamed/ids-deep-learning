# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 20:43:01 2022

@author: bauhaus
"""

from numpy import savetxt
import pandas as pd
import numpy as np 

###############################################################################
#Data Pre-processing
df = pd.read_csv("ALLdata.csv", on_bad_lines='skip')


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

df['proto'] = encoder.fit_transform(df['proto'])
df['service'] = encoder.fit_transform(df['service'])
df['state'] = encoder.fit_transform(df['state'])

#df.to_csv('ProcessedData.csv', index=False )

###############################################################################
# Separating Your Training and Testing Datasets

from sklearn.model_selection import train_test_split


X = df[['id','dur','proto','service','state','spkts','dpkts','sbytes','dbytes'
       ,'rate','sttl','dttl','sload','dload','sloss','dloss','sinpkt'
       ,'dinpkt','sjit','djit','swin','stcpb','dtcpb','dwin','tcprtt','synack','ackdat'
       ,'smean','dmean','trans_depth','response_body_len','ct_srv_src','ct_state_ttl',
       'ct_dst_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','is_ftp_login'
       ,'ct_ftp_cmd','ct_flw_http_mthd','ct_src_ltm','ct_srv_dst','is_sm_ips_ports']].values


y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

###############################################################################
#Transforming the Data
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

###############################################################################
#Building the Deep artificial Neural Network
from keras.models import Sequential
from keras.layers import Dense ,Dropout

classifier = Sequential()
classifier.add(Dense(40, kernel_initializer = "uniform",activation = "relu", input_dim=43))
classifier.add(Dense(20, kernel_initializer = "uniform",activation = "relu"))
classifier.add(Dropout(rate = 0.1))
classifier.add(Dense(10, kernel_initializer = "uniform",activation = "relu"))
classifier.add(Dense(1, kernel_initializer = "uniform",activation = "sigmoid"))
classifier.compile(optimizer= "RMSprop",loss = "binary_crossentropy",metrics = ["accuracy"])

classifier.fit(X_train, y_train, batch_size = 10, epochs = 1)

###############################################################################
#Running Predictions and evaluation on the Test Set


acc=classifier.evaluate(X_test, y_test, batch_size=10 ,verbose=1)

y_pred = classifier.predict(X_test)

print(y_pred)

###############################################################################
# Checking the Confusion Matrix 

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)
print(cm)

print(f1_score(y_test, y_pred))

###############################################################################
# Drow neural network

from keras.utils.vis_utils import plot_model

plot_model(classifier, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

###############################################################################
# save model

classifier.save('Model.h5')






