# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 20:43:01 2022

@author: bauhaus
"""

import pandas as pd
import numpy as np 
from numpy import savetxt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix


import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import model_from_yaml

###############################################################################

#Data Pre-processing

df = pd.read_csv("2222.csv", on_bad_lines='skip')

pd.set_option('display.max_columns', None)

encoder = LabelEncoder()

df['proto'] = encoder.fit_transform(df['proto'])
df['service'] = encoder.fit_transform(df['service'])
df['state'] = encoder.fit_transform(df['state'])



df.drop(['attack_cat'], axis = 1, inplace = True) 


print(df.dtypes)


print(df.head())
print(df.tail())

###############################################################################
# Separating Your Training and Testing Datasets



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

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

###############################################################################
#Building the Deep Neural Network

classifier = Sequential()
classifier.add(Dense(40, kernel_initializer = "uniform",activation = "relu", input_dim=43))
classifier.add(Dense(1, kernel_initializer = "uniform",activation = "sigmoid"))
classifier.compile(optimizer= "Adam",loss = "binary_crossentropy",metrics = ["accuracy"])

classifier.fit(X_train, y_train, batch_size = 10, epochs = 50)

###############################################################################
#Running Predictions and evaluation on the Test Set


acc=classifier.evaluate(X_test, y_test, batch_size=10 ,verbose=1)

#print("%s: %.2f%%" % (classifier.metrics_names[1], score[1]*100))
y=classifier.predict([[1,0.121478,113,0,3,6,4,258,172,74.08749,252,254,14158.94238,8495.365234,0,0,24.2956,8.375,30.177547,11.830604,255,621772692,2202533631,255,0,0,0,43,43,0,0,1,0,1,1,1,1,0,0,0,1,1,0]])
print("****************test")
print(y)


y_pred = classifier.predict(X_test)
print(y_pred)

print(classifier.summary())
from keras.utils.vis_utils import plot_model
plot_model(classifier, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#savetxt('result.csv', y_pred , delimiter=',')


print('************************************************************************')


y_pred = (y_pred > 0.5)



yhat = classifier.predict(X_test)
# round probabilities to class labels
yhat = yhat.round()
print(yhat)
"""
savetxt('20.csv', X_test, delimiter=',')
savetxt('21.csv', yhat, delimiter=',')
"""
###############################################################################
# Checking the Confusion Matrix


cm = confusion_matrix(y_test, y_pred)
"""
plt.figure()
sns.heatmap(pd.crosstab(y_test, y_pred) , annot=True , fmt='d')
plt.xlabel('target')
plt.ylabel('outcome')
plt.show()
"""
###############################################################################
# save model

#classifier.save_weights('Modelw.h5')

