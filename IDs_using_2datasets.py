# -*- coding: utf-8 -*-
"""
Created on Sat May  7 20:52:31 2022

@author: bauhaus
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


import keras
from keras.models import Sequential
from keras.layers import Dense

from sklearn.metrics import confusion_matrix

###############################################################################

#Data Pre-processing

train = pd.read_csv("Train.csv", on_bad_lines='skip')
test = pd.read_csv("test.csv", on_bad_lines='skip')



encoder = LabelEncoder()

train ['proto'] = encoder.fit_transform(train ['proto'])
train ['service'] = encoder.fit_transform(train ['service'])
train ['state'] = encoder.fit_transform(train ['state'])

test['proto'] = encoder.fit_transform(test['proto'])
test['service'] = encoder.fit_transform(test['service'])
test['state'] = encoder.fit_transform(test['state'])






train .drop(['attack_cat'], axis = 1, inplace = True) 
test.drop(['attack_cat'], axis = 1, inplace = True) 


print(train .dtypes)
print(test.dtypes)

print(train .head())
print(test.head())

###############################################################################
# Separating Your Training and Testing Datasets



X = train [['id','dur','proto','service','state','spkts','dpkts','sbytes','dbytes'
       ,'rate','sttl','dttl','sload','dload','sloss','dloss','sinpkt'
       ,'dinpkt','sjit','djit','swin','stcpb','dtcpb','dwin','tcprtt','synack','ackdat'
       ,'smean','dmean','trans_depth','response_body_len','ct_srv_src','ct_state_ttl',
       'ct_dst_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','is_ftp_login'
       ,'ct_ftp_cmd','ct_flw_http_mthd','ct_src_ltm','ct_srv_dst','is_sm_ips_ports']].values

y = train ['label'].values


Xt = test[['id','dur','proto','service','state','spkts','dpkts','sbytes','dbytes'
       ,'rate','sttl','dttl','sload','dload','sloss','dloss','sinpkt'
       ,'dinpkt','sjit','djit','swin','stcpb','dtcpb','dwin','tcprtt','synack','ackdat'
       ,'smean','dmean','trans_depth','response_body_len','ct_srv_src','ct_state_ttl',
       'ct_dst_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','is_ftp_login'
       ,'ct_ftp_cmd','ct_flw_http_mthd','ct_src_ltm','ct_srv_dst','is_sm_ips_ports']].values


yt = test['label'].values

X_train, y_train = X, y
X_test, y_test = Xt, yt



###############################################################################
#Transforming the Data

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

###############################################################################
#Building the Artificial Neural Network

classifier = Sequential()
classifier.add(Dense(2, kernel_initializer = "uniform",activation = "relu", input_dim=43))
classifier.add(Dense(1, kernel_initializer = "uniform",activation = "sigmoid"))
classifier.compile(optimizer= "adam",loss = "binary_crossentropy",metrics = ["accuracy"])

classifier.fit(X_train, y_train, batch_size = 10, epochs = 1)

###############################################################################
#Running Predictions on the Test Set

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)

###############################################################################
# Checking the Confusion Matrix


cm = confusion_matrix(y_test, y_pred)

